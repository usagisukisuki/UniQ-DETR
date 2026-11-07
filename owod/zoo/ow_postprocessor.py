"""Open-world post-processor.

Per query q with class probs p(c|q)=sigmoid(logit_c) and objectness o(q):
  known c : s_c   = o(q) * p(c|q)          (known_score='obj_cls')  or  p(c|q) ('cls')
  unknown : s_unk = unk_score(q)
     'c1'     o(q) * (1 - max_c p(c|q))                    (plan C1)
     'maha'   o(q)                                         (pure Mahalanobis objectness)
     'msp'    1 - max_c p(c|q)                             (post-hoc baseline, no objectness)
     'energy' sigmoid(-T * logsumexp(logit / T))           (post-hoc baseline, no objectness)
     'learned' o(q) * sigmoid(unk_logit)                   (PROB-style learned unknown logit)
     'learned_q' o(q) * sigmoid(unk_logit) * quality(q)    (F1: x class-agnostic localisation quality)
     'su_q'    sigmoid(unk_logit) * quality(q)             (F1 without o)
     'learned_qr'   G: o * sigmoid(u) * q * sigmoid(r), r = one-to-one 'primary query' head (NMS-free unknowns)
     'v' / 'su_v' / 'learned_v' / 'learned_q_v'   F3 'unlabelled object' head sigmoid(v), alone or
              multiplied into sigmoid(u) / o*sigmoid(u) / o*sigmoid(u)*q
     'cls_last' the classifier's last logit is the 'unknown' class: known = p[:, :-1], unknown = p[:, -1]
              (OW-DETR / CAT / RandBox; num_classes includes the unknown class)
     'none'   no unknown output (closed set)
Scores of shape [Q, K(+1)] are flattened and the top `num_top_queries` (default 100, as
in PROB) are returned. Unknown label index == num_classes.

Known/unknown balance in the shared top-100 (docs/PART1_RESULTS.md):
  unk_weight  s_unk is multiplied by this factor before ranking (1.0 = off)
  unk_slots   at most this many of the top-100 are unknown; the known top-(100 - n) and the
              unknown top-n are selected separately (None = off, one shared ranking)
  unk_dup_iou / unk_dup_tau
              duplicate suppression: the unknown score of a query is zeroed if its box overlaps
              (IoU >= unk_dup_iou) another query whose max known p >= unk_dup_tau, i.e. it is a
              duplicate of a known detection (None = off; docs/NEXT_STEPS.md, F1)
  unk_dup_area_ratio  (optional) additionally require the two boxes' area ratio <= this (a duplicate has
              about the same size; an unknown object lying on a known one, e.g. a bowl on a table, does not)
  unk_dup_cls_p       (optional) additionally require p(q, c') >= this, where c' is the other query's
              class (a duplicate still scores the known object's class)
  unk_dup_tau_hi      (optional) two-tier rule: if the other query's max p >= this (a confident, likely
              correct known detection) the duplicate is suppressed regardless of unk_dup_cls_p; the class
              condition only applies to moderately confident known detections (possible misclassified unknowns)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from src.core import register

__all__ = ['OWPostProcessor']


@register()
class OWPostProcessor(nn.Module):
    __share__ = ['num_classes', ]

    def __init__(self, num_classes=20, num_top_queries=100, unk_score='c1', known_score='obj_cls',
                 energy_temperature=1.0, unk_weight=1.0, unk_slots=None, unk_dup_iou=None, unk_dup_tau=0.3,
                 unk_dup_area_ratio=None, unk_dup_cls_p=None, unk_dup_tau_hi=None, unk_v_gamma=1.0,
                 known_slots=None, known_scale=1.0, nms_iou=None, nms_pre=10000, unk_self_iou=None, unk_r_gamma=1.0,
                 unk_maxp_beta=0.0, unk_punk_delta=0.0, unk_o_exp=1.0, unk_u_exp=1.0, unk_q_exp=1.0,
                 unk_notk_beta=0.0, unk_area_exp=0.0, known_r_gamma=0.0, one_label=False):
        super().__init__()
        assert unk_score in ('c1', 'maha', 'msp', 'energy', 'learned', 'learned_q', 'su_q',
                             'v', 'su_v', 'learned_v', 'learned_q_v', 'learned_qr', 'cls_last', 'cls_last_closed', 'none')
        assert known_score in ('obj_cls', 'cls')
        self.num_classes = int(num_classes)
        self.num_top_queries = num_top_queries
        self.unk_score = unk_score
        self.known_score = known_score
        self.energy_temperature = energy_temperature
        self.unk_weight = unk_weight
        self.unk_slots = unk_slots
        self.unk_dup_iou = unk_dup_iou
        self.unk_dup_tau = unk_dup_tau
        self.unk_dup_area_ratio = unk_dup_area_ratio
        self.unk_dup_cls_p = unk_dup_cls_p
        self.unk_dup_tau_hi = unk_dup_tau_hi
        self.unk_v_gamma = unk_v_gamma        # F3: sigmoid(v) ** gamma (gamma < 1 softens the v factor)
        # baselines (docs/BASELINES_RTDETR.md): cls_last keeps only the first known_slots logits as known classes
        # (OrthogonalDet has 80 slots), known scores x known_scale (OrthogonalDet 0.75), and class-wise NMS before
        # the final top-k (RandBox / OrthogonalDet: one-to-many matching, NMS 0.6 over the top nms_pre pairs)
        self.known_slots = known_slots
        self.known_scale = known_scale
        self.nms_iou = nms_iou
        self.nms_pre = nms_pre
        # NMS-free unknown dedup (docs/BASELINES_RESULTS.md §11): one matrix op over the queries (Fast-NMS style, no
        # sequential loop) -- a query's unknown score is zeroed if another query with a higher unknown score overlaps
        # it with IoU >= unk_self_iou. Known scores are untouched.
        self.unk_self_iou = unk_self_iou
        self.unk_r_gamma = unk_r_gamma        # G: sigmoid(r) ** gamma (gamma < 1 softens the one-to-one factor)
        # G4-2 (analysis/G4_PLAN.md): learned_qr unknown score x (max known p)^beta x p_unk^delta, where p_unk is the
        # extra 'unknown' logit after the known_slots known ones (OW-DETR signal); 0 = factor off
        self.unk_maxp_beta = unk_maxp_beta
        self.unk_punk_delta = unk_punk_delta
        # G4 U-Recall: x (1 - max known p)^beta, to push down unknown outputs that duplicate a known object; 0 = off
        self.unk_notk_beta = unk_notk_beta
        # G4 U-Recall: unknown score x (normalised box area w*h)^a, any unk_score (also the baselines); box area separates
        # unknown objects well (AUROC 0.795, analysis/G4_PLAN.md 6.1); 0 = off
        self.unk_area_exp = unk_area_exp
        self.unk_o_exp, self.unk_u_exp, self.unk_q_exp = unk_o_exp, unk_u_exp, unk_q_exp   # learned_qr exponents
        # A-OSE (analysis/G4_PLAN.md, G4-8): known scores x sigmoid(r)^known_r_gamma pushes down the known outputs of
        # duplicate queries (0 = off); one_label keeps only each query's highest final score (after unk_weight), so a
        # query that reports an unknown object does not also report a low-score known class
        self.known_r_gamma = known_r_gamma
        self.one_label = one_label
        self.deploy_mode = False

    def extra_repr(self):
        return (f'num_classes={self.num_classes}, num_top_queries={self.num_top_queries}, '
                f'unk_score={self.unk_score}, known_score={self.known_score}, '
                f'unk_weight={self.unk_weight}, unk_slots={self.unk_slots}')

    def scores(self, outputs):
        logits = outputs['pred_logits'].float()
        p = logits.sigmoid()
        maxp = p.max(-1).values
        needs_obj = self.known_score == 'obj_cls' or self.unk_score in ('c1', 'maha', 'learned', 'learned_q',
                                                                         'learned_v', 'learned_q_v', 'learned_qr')
        o = outputs.get('pred_objectness', None)
        if needs_obj:
            assert o is not None, f'{self.known_score}/{self.unk_score} need an objectness head'
            o = o.float()
        k = self.known_slots if self.known_slots is not None else p.shape[-1] - 1
        if self.unk_score == 'cls_last':
            return torch.cat([p[..., :k] * self.known_scale, p[..., -1:]], dim=-1)
        if self.unk_score == 'cls_last_closed':           # closed-set view of a model with an unknown class
            return p[..., :k]
        p_unk = p[..., k] if (self.known_slots is not None and p.shape[-1] > k) else None
        if self.known_slots is not None:      # classifier carries extra (e.g. 'unknown') logits after the known ones
            p = p[..., :k]
            maxp = p.max(-1).values
        known = p * o.unsqueeze(-1) if self.known_score == 'obj_cls' else p
        if self.known_r_gamma:
            known = known * outputs['pred_primary_logits'].float().sigmoid().unsqueeze(-1) ** self.known_r_gamma
        if self.unk_score == 'none':
            return known
        if self.unk_score == 'c1':
            unk = o * (1 - maxp)
        elif self.unk_score == 'maha':
            unk = o
        elif self.unk_score == 'learned':
            assert 'pred_unk_logits' in outputs, "unk_score='learned' needs use_unk_logit=True"
            unk = o * outputs['pred_unk_logits'].float().sigmoid()
        elif self.unk_score in ('v', 'su_v', 'learned_v', 'learned_q_v'):
            unk = outputs['pred_unkobj_logits'].float().sigmoid() ** self.unk_v_gamma
            if self.unk_score != 'v':
                unk = unk * outputs['pred_unk_logits'].float().sigmoid()
            if self.unk_score in ('learned_v', 'learned_q_v'):
                unk = unk * o
            if self.unk_score == 'learned_q_v':
                unk = unk * outputs['pred_quality_logits'].float().sigmoid()
        elif self.unk_score == 'learned_qr':
            unk = (o ** self.unk_o_exp * outputs['pred_unk_logits'].float().sigmoid() ** self.unk_u_exp
                   * outputs['pred_quality_logits'].float().sigmoid() ** self.unk_q_exp
                   * outputs['pred_primary_logits'].float().sigmoid() ** self.unk_r_gamma)
            if self.unk_maxp_beta:
                unk = unk * maxp ** self.unk_maxp_beta
            if self.unk_notk_beta:
                unk = unk * (1 - maxp).clamp(min=0) ** self.unk_notk_beta
            if self.unk_punk_delta:
                assert p_unk is not None, 'unk_punk_delta needs an extra unknown logit (known_slots)'
                unk = unk * p_unk ** self.unk_punk_delta
        elif self.unk_score in ('learned_q', 'su_q'):
            q = outputs['pred_quality_logits'].float().sigmoid()
            su = outputs['pred_unk_logits'].float().sigmoid()
            unk = su * q * (o if self.unk_score == 'learned_q' else 1.)
        elif self.unk_score == 'msp':
            unk = 1 - maxp
        else:
            T = self.energy_temperature
            unk = torch.sigmoid(-T * torch.logsumexp(logits / T, dim=-1))
        return torch.cat([known, unk.unsqueeze(-1)], dim=-1)

    def forward(self, outputs, orig_target_sizes: torch.Tensor):
        boxes = torchvision.ops.box_convert(outputs['pred_boxes'], in_fmt='cxcywh', out_fmt='xyxy')
        boxes = boxes * orig_target_sizes.repeat(1, 2).unsqueeze(1)
        scores = self.scores(outputs)
        has_unk = self.unk_score not in ('none', 'cls_last_closed')
        if has_unk and self.unk_area_exp:
            area = (outputs['pred_boxes'][..., 2].float() * outputs['pred_boxes'][..., 3].float()).clamp(min=1e-6)
            scores = torch.cat([scores[..., :-1], scores[..., -1:] * area.unsqueeze(-1) ** self.unk_area_exp], dim=-1)
        if has_unk and self.unk_dup_iou is not None:
            p = outputs['pred_logits'].float().sigmoid()
            if self.known_slots is not None:
                p = p[..., :self.known_slots]
            pmax, pcls = p.max(-1)                                                   # [B, Q]
            xyxy = torchvision.ops.box_convert(outputs['pred_boxes'].float(), in_fmt='cxcywh', out_fmt='xyxy')
            ov = torch.stack([torchvision.ops.box_iou(b, b) for b in xyxy]) >= self.unk_dup_iou   # [B, Q, Q']
            ov &= ~torch.eye(ov.shape[1], dtype=torch.bool, device=ov.device)
            if self.unk_dup_area_ratio is not None:
                area = outputs['pred_boxes'][..., 2].float() * outputs['pred_boxes'][..., 3].float()
                ratio = area.unsqueeze(2) / area.unsqueeze(1).clamp(min=1e-9)
                ov &= (ratio <= self.unk_dup_area_ratio) & (ratio >= 1. / self.unk_dup_area_ratio)
            if self.unk_dup_cls_p is not None:
                # p(q, class of q') >= rho
                p_other = p.gather(2, pcls.unsqueeze(1).expand(-1, p.shape[1], -1))    # [B, Q, Q']
                cls_ok = p_other >= self.unk_dup_cls_p
                if self.unk_dup_tau_hi is not None:
                    cls_ok |= (pmax >= self.unk_dup_tau_hi).unsqueeze(1)
                ov &= cls_ok
            dup = (ov & (pmax >= self.unk_dup_tau).unsqueeze(1)).any(-1)
            scores = torch.cat([scores[..., :-1], scores[..., -1:] * (~dup).unsqueeze(-1)], dim=-1)
        if has_unk and self.unk_self_iou is not None:
            u = scores[..., -1]                                                      # [B, Q]
            xyxy = torchvision.ops.box_convert(outputs['pred_boxes'].float(), in_fmt='cxcywh', out_fmt='xyxy')
            ov = torch.stack([torchvision.ops.box_iou(b, b) for b in xyxy]) >= self.unk_self_iou   # [B, Q, Q']
            higher = (u.unsqueeze(1) > u.unsqueeze(2)) | ((u.unsqueeze(1) == u.unsqueeze(2)) &
                     torch.ones_like(ov).triu(1).transpose(1, 2))                    # q' ranks above q
            sup = (ov & higher).any(-1)
            scores = torch.cat([scores[..., :-1], scores[..., -1:] * (~sup).unsqueeze(-1)], dim=-1)
        if has_unk and self.unk_weight != 1.0:
            scores = torch.cat([scores[..., :-1], scores[..., -1:] * self.unk_weight], dim=-1)
        if self.one_label:
            scores = scores * (scores == scores.max(-1, keepdim=True).values)
        ncls = scores.shape[-1]
        if has_unk and self.unk_slots is not None:
            nq, nk = scores.shape[1], ncls - 1
            n_unk = min(self.unk_slots, self.num_top_queries, nq)
            k_top, k_idx = torch.topk(scores[..., :-1].flatten(1), self.num_top_queries - n_unk, dim=-1)
            u_top, u_q = torch.topk(scores[..., -1], n_unk, dim=-1)
            top = torch.cat([k_top, u_top], dim=1)
            labels = torch.cat([k_idx % nk, torch.full_like(u_q, nk)], dim=1)
            qidx = torch.cat([k_idx // nk, u_q], dim=1)
            top, order = top.sort(dim=1, descending=True)
            labels, qidx = labels.gather(1, order), qidx.gather(1, order)
        elif self.nms_iou is not None:
            # variable number of detections per image (<= num_top_queries) after class-wise NMS
            k0 = min(self.nms_pre, scores.shape[1] * ncls)
            top0, idx0 = torch.topk(scores.flatten(1), k0, dim=-1)
            lab0, q0 = idx0 % ncls, idx0 // ncls
            res = []
            for b in range(scores.shape[0]):
                bx = boxes[b, q0[b]]
                keep = torchvision.ops.batched_nms(bx.float(), top0[b].float(), lab0[b], self.nms_iou)[:self.num_top_queries]
                res.append(dict(labels=lab0[b, keep], boxes=bx[keep], scores=top0[b, keep], query_index=q0[b, keep]))
            return res
        else:
            k = min(self.num_top_queries, scores.shape[1] * ncls)
            top, index = torch.topk(scores.flatten(1), k, dim=-1)
            labels = index % ncls
            qidx = index // ncls
        boxes = boxes.gather(1, qidx.unsqueeze(-1).repeat(1, 1, 4))
        if self.deploy_mode:
            return labels, boxes, top
        return [dict(labels=l, boxes=b, scores=s, query_index=q) for l, b, s, q in zip(labels, boxes, top, qidx)]

    def deploy(self):
        self.eval()
        self.deploy_mode = True
        return self
