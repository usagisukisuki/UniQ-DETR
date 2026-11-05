"""Open-world criterion for RT-DETRv2.

C2: loss_obj = sum_{matched} max(d2(q_i)/D, obj_min) / num_boxes   (positive-only; no
    loss on unmatched queries). The plan's L = 1/2 d2 is rescaled by 1/D so that
    lambda_obj=0.3 is in the same range as PROB (8e-4 on raw d2, D=256 -> ~0.2).
    obj_min = -log(0.9) follows PROB (stop pulling once o(q) would exceed 0.9 at tau=1).
    E-step statistics (obj_stats):
      'matched' (plan C2 literal): (mu, Sigma) from matched embeddings only. With a positive-only
                loss this saturates: the shared decoder drags *all* queries toward mu, background
                ends up inside the foreground Gaussian (E2 v1: mean o(q)=0.84, 97% of queries with
                d2/D<=1, Known AP50 5 at ep10).
      'all'     (PROB-style reference): (mu, Sigma) from all decoder queries of the layer
                (matched + unmatched; DN queries are already split off). The loss is still applied
                to matched queries only, so unmatched queries get no gradient (H1 kept), but they
                now define the reference: background sits at d2/D~1, matched queries are pulled
                below it. The encoder head (C4) always uses matched tokens.
C1 (learned unknown, PROB-style): loss_unk = w_empty * sum_q focal(unk_logit_q, 1[q unmatched]) / num_boxes,
    i.e. PROB's extra 'unknown' class (index 80) whose target is 1 for every unmatched query and
    whose focal loss is down-weighted by empty_weight=0.1. Inference: s_unk = o(q) * sigmoid(unk_logit).
F2 (unk_neg_iou > 0): unmatched queries whose box overlaps a known GT with IoU >= unk_neg_iou are
    duplicates / fragments of a known object, so they get target 0 (not 1) in loss_unk.
F1 loss_quality: class-agnostic localisation quality (OLN, Kim et al. 2022). Queries whose box has
    IoU >= quality_min_iou with some known GT are trained (BCE, soft target) to predict that IoU;
    all other queries get no loss (no background negatives, so unknown objects are not suppressed).
F3 loss_unkobj (targets carry 'heldout_mask', set by the decoder): objects of the held-out classes are
    removed from the targets used by every other loss (so they look exactly like real unknowns), and an
    'unlabelled object' logit v is trained with focal loss: positive = query box IoU >= unkobj_pos_iou with
    a held-out object, negative = IoU < unkobj_neg_iou with every GT (known and held-out), others ignored.
OW-DETR (attn_pseudo_k > 0; docs/baselines/OWDETR_SPEC.md): from epoch attn_pseudo_start, per output layer the
    attn_pseudo_k unmatched queries with the highest mean backbone-C4 activation inside their box (outputs['attn_map'])
    become 'unknown' targets (label num_classes - 1, box = detached predicted box so the VFL IoU target is 1). They
    enter only the classification and objectness ('ncobj') losses; box losses and num_boxes use GT only.
    loss_ncobj: sigmoid focal on pred_nc_logits, target 1 for matched + pseudo-unknown queries (OW-DETR 'NC' branch),
    weight forced to 0 before attn_pseudo_start (OW-DETR nc_epoch).
B (pseudo_score = 'unk'; docs/BASELINES_RESULTS.md §8): as OW-DETR, but the unmatched queries are ranked by the model's
    own detached unknown score o * sigmoid(u) * sigmoid(q) of the same layer instead of the C4 activation, and queries
    with IoU >= pseudo_known_iou to a known GT (duplicates / fragments of known objects) are not eligible.
G loss_primary (NMS-free unknowns, docs/BASELINES_RESULTS.md §11.5): class-agnostic one-to-one head r. Target 1 for
    the Hungarian-matched query of each known GT, 0 for the other queries overlapping a known GT with IoU >=
    primary_dup_iou (its duplicates); queries far from every GT get no loss, so r does not learn 'background' and
    does not suppress unknown objects (the failure of F2). r learns 'the one query that represents its object' from
    many known classes and is meant to transfer to unknown objects, replacing NMS for the unknown score.
    G2 (primary_bg_frac > 0): additionally, among the queries with IoU < primary_bg_iou to every known GT, those whose
    objectness o is in the image's lowest primary_bg_frac get target 0 (confident background; low-o keeps likely
    unknown objects, which tend to have higher o, out of the negatives).
B2 (NMS-free unknowns, docs/BASELINES_RESULTS.md §11): pseudo positives are picked greedily so that no two overlap with
    IoU >= pseudo_excl_iou (one positive per pseudo-object), and in loss_unk the other unmatched queries overlapping a
    pseudo positive with IoU >= pseudo_dup_neg_iou get target 0 -- a one-to-one assignment for unknowns, so the
    unknown score itself learns to fire once per object, as the known classes do under Hungarian matching.
G4-1 (pseudo_score = 'ext'; analysis/G4_PLAN.md): the pseudo-unknowns come from offline teacher boxes in the targets
    (label num_classes - 1, tools/make_pseudo_unk.py) instead of being picked online. They are split off before matching,
    so every known-GT loss, num_boxes, r and q see known GT only. From attn_pseudo_start, per output layer each pseudo box
    takes the one unmatched query with the highest IoU (>= pseudo_ext_iou, a query serves at most one box) as an
    'unknown' positive for the classification and objectness losses, with the pseudo box as its box (VFL target = IoU).
G4-3 (outputs['o2m_outputs'], o2m_k > 0; analysis/G4_PLAN.md): H-DETR hybrid matching. A training-only group of decoder
    queries is Hungarian-matched to the known GT repeated o2m_k times (o2m_k positives per object); only the known-class
    VFL (unknown column excluded) and box losses are applied, per layer, weighted by o2m_weight. No unknown head,
    pseudo-label, r or q loss touches this group, so it does not conflict with r's 'duplicates are 0'.
CAT (cat_ss = True; docs/baselines/CAT_SPEC.md): the pseudo-unknown score is the self-adaptive combination
    S_i = norm(s_o,i)^W_m * (max_j IoU(SS_j, box_i))^W_I of the OW-DETR attention score s_o (min-max normalised over the
    image's unmatched queries) and the overlap with precomputed selective-search boxes (targets[b]['ss_boxes']).
    W_m, W_I start at 0.8 / 0.2 and are updated every cat_tb iterations from the loss trend (paper Eq. 6-8):
    dl = (weighted mean of the last 20 total losses) / (weighted mean of the 20 before), dw = 0.5*sigmoid(dl-1) if
    dl > 1 else -dl/3, W_m += dw*W_m, W_I -= dw*W_I, renormalised to sum 1.
RandBox (randbox_unk_k > 0; docs/baselines/RANDBOX_SPEC.md): with the one-to-many SimOTA matcher, per output layer the
    randbox_unk_k queries with the highest matching score sum_c sigmoid(logit_c) (excluding each GT's best query) are
    'unknown'; loss_randbox_unk is a second sigmoid focal loss over all queries with one-hot 'unknown' on them and zeros
    elsewhere, normalised by their number, active after randbox_unk_start iterations. Losses are normalised by the
    number of matched queries (one-to-many), as in RandBox.
C3: among unmatched queries, the k_bg with the highest objectness (smallest d2) are
    excluded from the background (all-zero target) classification loss.
"""

import collections
import copy
import math

import torch
import torch.nn.functional as F
import torchvision

from src.core import register
from src.misc.dist_utils import get_world_size, is_dist_available_and_initialized
from src.zoo.rtdetr.box_ops import box_cxcywh_to_xyxy, box_iou
from src.zoo.rtdetr.rtdetrv2_criterion import RTDETRCriterionv2

__all__ = ['OWCriterion']


@register()
class OWCriterion(RTDETRCriterionv2):
    __share__ = ['num_classes', ]
    __inject__ = ['matcher', ]

    def __init__(self,
                 matcher,
                 weight_dict,
                 losses,
                 alpha=0.2,
                 gamma=2.0,
                 num_classes=80,
                 boxes_weight_format=None,
                 share_matched_indices=False,
                 # --- open-world ---
                 obj_min=-math.log(0.9),
                 bg_ignore_topk=0,          # C3 k_bg (0 = off)
                 bg_ignore_enc=True,        # also apply C3 to the encoder top-K outputs
                 obj_stats='matched',       # E-step statistics: 'matched' (plan) | 'all' decoder queries
                 unk_empty_weight=0.1,      # PROB empty_weight on the learned unknown logit
                 unk_alpha=0.25,            # PROB focal_alpha
                 unk_neg_iou=0.0,           # F2: unmatched queries with IoU >= this to a known GT -> unk target 0 (0 = off)
                 quality_min_iou=0.3,       # F1: queries with IoU >= this to a known GT get a quality target
                 pseudo_primary=False,      # G4-6: the one-to-one head r also learns the pseudo-unknown boxes
                 primary_dup_iou=0.5,       # G: unmatched queries with IoU >= this to a known GT -> primary target 0
                 primary_bg_frac=0.0,       # G2: lowest-objectness fraction of far-from-GT queries -> target 0 (0 = off)
                 primary_bg_iou=0.3,        # G2: 'far from GT' = max IoU < this
                 unkobj_pos_iou=0.5,        # F3
                 unkobj_neg_iou=0.3,        # F3
                 unkobj_pos_top1=False,     # F3c: positive = only the best-overlapping query of each held-out object
                 unkobj_dup_neg=False,      # F3c: duplicates of known objects (IoU >= 0.5 to a known GT, not its
                                            #      best-overlapping query) are negatives for v
                 attn_pseudo_k=0,           # OW-DETR: pseudo-unknowns per image and layer (0 = off)
                 attn_pseudo_start=9,       # OW-DETR: first epoch with pseudo-labels and the objectness loss
                 pseudo_score='attn',       # 'attn' (OW-DETR, C4 activation) | 'unk' (B, model's own unknown score)
                 pseudo_known_iou=0.0,      # B: unmatched queries with IoU >= this to a known GT are not eligible (0 = off)
                 pseudo_excl_iou=0.0,       # B2: pseudo positives pairwise IoU < this (0 = off)
                 pseudo_dup_neg_iou=0.0,    # B2: duplicates of a pseudo positive -> unk target 0 (0 = off)
                 pseudo_ext_iou=0.5,        # G4-1: min IoU between a pseudo box and its positive query
                 o2m_k=0,                   # G4-3: GT repeats for the one-to-many group (0 = off)
                 o2m_weight=1.0,            # G4-3: weight of the one-to-many losses
                 o2m_known_slots=None,      # G4-3: classification columns used by the one-to-many VFL (None = all)
                 cat_ss=False,              # CAT: self-adaptive model-driven + selective-search pseudo-label score
                 cat_tb=40,                 # CAT: iterations between W_m / W_I updates
                 randbox_unk_k=0,           # RandBox / OrthogonalDet: unknown queries per image and layer (0 = off)
                 randbox_unk_start=500,     # RandBox: iterations before the unknown loss is switched on
                 decorr=False,              # OrthogonalDet: prediction decorrelation loss (needs pred_cls_prob / pred_pobj)
                 ):
        super().__init__(matcher, weight_dict, losses, alpha=alpha, gamma=gamma, num_classes=num_classes,
                         boxes_weight_format=boxes_weight_format, share_matched_indices=share_matched_indices)
        self.obj_min = obj_min
        self.bg_ignore_topk = bg_ignore_topk
        self.bg_ignore_enc = bg_ignore_enc
        assert obj_stats in ('all', 'matched')
        self.obj_stats = obj_stats
        self.unk_empty_weight = unk_empty_weight
        self.unk_alpha = unk_alpha
        self.unk_neg_iou = unk_neg_iou
        self.quality_min_iou = quality_min_iou
        self.primary_dup_iou = primary_dup_iou
        self.pseudo_primary = pseudo_primary
        self.primary_bg_frac = primary_bg_frac
        self.primary_bg_iou = primary_bg_iou
        self._obj_heads = None
        self.unkobj_pos_iou = unkobj_pos_iou
        self.unkobj_neg_iou = unkobj_neg_iou
        self.unkobj_pos_top1 = unkobj_pos_top1
        self.unkobj_dup_neg = unkobj_dup_neg
        self.attn_pseudo_k = attn_pseudo_k
        self.attn_pseudo_start = attn_pseudo_start
        assert pseudo_score in ('attn', 'unk', 'ext')
        self.pseudo_score = pseudo_score
        self.pseudo_known_iou = pseudo_known_iou
        self.pseudo_excl_iou = pseudo_excl_iou
        self.pseudo_dup_neg_iou = pseudo_dup_neg_iou
        self.pseudo_ext_iou = pseudo_ext_iou
        self.o2m_k = o2m_k
        self.o2m_weight = o2m_weight
        self.o2m_known_slots = o2m_known_slots
        self._ext_boxes = None
        self.cat_ss = cat_ss
        self.cat_tb = cat_tb
        self.cat_w = [0.8, 0.2]                      # W_m (model-driven), W_I (input-driven)
        self._loss_mem = collections.deque(maxlen=40)
        self._iter = 0
        self.randbox_unk_k = randbox_unk_k
        self.randbox_unk_start = randbox_unk_start
        self.decorr = decorr
        self._train_iter = 0
        self._epoch = 0

    # ------------------------------------------------------------------ label losses with ignore mask
    def _keep(self, outputs, ignore):
        if ignore is None:
            return None
        return (~ignore).to(outputs['pred_logits'].dtype).unsqueeze(-1)

    def loss_labels_focal(self, outputs, targets, indices, num_boxes, ignore=None, **kw):
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1].to(src_logits.dtype)
        loss = torchvision.ops.sigmoid_focal_loss(src_logits, target, self.alpha, self.gamma, reduction='none')
        keep = self._keep(outputs, ignore)
        if keep is not None:
            loss = loss * keep
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {'loss_focal': loss}

    def loss_labels_vfl(self, outputs, targets, indices, num_boxes, values=None, ignore=None, **kw):
        idx = self._get_src_permutation_idx(indices)
        if values is None:
            src_boxes = outputs['pred_boxes'][idx]
            target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
            ious, _ = box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
            ious = torch.diag(ious).detach()
        else:
            ious = values
        src_logits = outputs['pred_logits']
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]
        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        target_score_o[idx] = ious.to(target_score_o.dtype)
        target_score = target_score_o.unsqueeze(-1) * target
        pred_score = F.sigmoid(src_logits).detach()
        weight = self.alpha * pred_score.pow(self.gamma) * (1 - target) + target_score
        keep = self._keep(outputs, ignore)
        if keep is not None:
            weight = weight * keep
        loss = F.binary_cross_entropy_with_logits(src_logits, target_score, weight=weight, reduction='none')
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {'loss_vfl': loss}

    def loss_boxes(self, outputs, targets, indices, num_boxes, boxes_weight=None, **kw):
        return super().loss_boxes(outputs, targets, indices, num_boxes, boxes_weight=boxes_weight)

    # ------------------------------------------------------------------ C2
    def loss_obj(self, outputs, targets, indices, num_boxes, **kw):
        idx = self._get_src_permutation_idx(indices)
        d2 = outputs['pred_obj'][idx]
        dim = outputs['pred_embed'].shape[-1]
        loss = torch.clamp(d2 / dim, min=self.obj_min).sum() / num_boxes
        return {'loss_obj': loss}

    def loss_unk(self, outputs, targets, indices, num_boxes, **kw):
        logits = outputs['pred_unk_logits']                       # [B, Q]
        target = torch.ones_like(logits)
        target[self._get_src_permutation_idx(indices)] = 0.
        if self.unk_neg_iou > 0:
            target = target * (self._max_iou_to_gt(outputs, targets) < self.unk_neg_iou).to(target.dtype)
        pseudo_ti = kw.get('pseudo_ti', None)
        if pseudo_ti is not None and self.pseudo_dup_neg_iou > 0:
            pb = box_cxcywh_to_xyxy(outputs['pred_boxes'].detach().float())
            unk = self.num_classes - 1
            for b, (t, (src, tgt)) in enumerate(zip(*pseudo_ti)):
                pq = src.to(pb.device)[t['labels'][tgt.to(t['labels'].device)] == unk]   # pseudo-positive queries of image b
                if len(pq) == 0:
                    continue
                dup = (box_iou(pb[b], pb[b, pq])[0] >= self.pseudo_dup_neg_iou).any(-1)
                dup[pq] = False
                target[b][dup] = 0.
        loss = torchvision.ops.sigmoid_focal_loss(logits, target, self.unk_alpha, self.gamma, reduction='sum')
        return {'loss_unk': self.unk_empty_weight * loss / num_boxes}

    @torch.no_grad()
    def _max_iou_to_gt(self, outputs, targets):
        """[B, Q] max IoU of each predicted box with the image's (known) GT boxes."""
        pb = box_cxcywh_to_xyxy(outputs['pred_boxes'].detach().float())
        out = torch.zeros(pb.shape[:2], device=pb.device)
        for b, t in enumerate(targets):
            if len(t['boxes']):
                out[b] = box_iou(pb[b], box_cxcywh_to_xyxy(t['boxes'].float()))[0].max(-1).values
        return out

    def loss_primary(self, outputs, targets, indices, num_boxes, **kw):
        logits = outputs['pred_primary_logits']                   # [B, Q]
        pos = torch.zeros_like(logits, dtype=torch.bool)
        pos[self._get_src_permutation_idx(indices)] = True
        miou = self._max_iou_to_gt(outputs, targets)
        use = pos | (miou >= self.primary_dup_iou)
        loss = F.binary_cross_entropy_with_logits(logits[use].float(), pos[use].float(), reduction='sum') / max(float(use.sum()), 1.)
        if self.primary_bg_frac > 0 and self._obj_heads is not None and 'obj_key' in outputs:
            # G2: confident background, normalised separately so it does not swamp the one-to-one term
            with torch.no_grad():
                o = self._obj_heads[outputs['obj_key']].objectness(outputs['pred_obj'].detach().float())   # [B, Q]
                far = (miou < self.primary_bg_iou) & ~pos
                bg = torch.zeros_like(far)
                for b in range(len(o)):
                    if far[b].any():
                        bg[b] = far[b] & (o[b] <= o[b][far[b]].quantile(self.primary_bg_frac))
            if bg.any():
                loss = loss + F.binary_cross_entropy_with_logits(logits[bg].float(), torch.zeros_like(logits[bg].float()))
        return {'loss_primary': loss}

    def loss_quality(self, outputs, targets, indices, num_boxes, **kw):
        logits = outputs['pred_quality_logits']                   # [B, Q]
        tgt = self._max_iou_to_gt(outputs, targets)
        m = tgt >= self.quality_min_iou
        if not m.any():
            return {'loss_quality': logits.sum() * 0.}
        loss = F.binary_cross_entropy_with_logits(logits[m].float(), tgt[m], reduction='sum')
        return {'loss_quality': loss / max(float(m.sum()), 1.)}

    def _loss_unkobj(self, outputs, held_boxes, all_boxes, known_boxes=None):
        logits = outputs['pred_unkobj_logits']                    # [B, Q]
        pb = box_cxcywh_to_xyxy(outputs['pred_boxes'].detach().float())
        tgt = torch.zeros_like(logits); use = torch.zeros_like(logits, dtype=torch.bool)
        n_pos = 0.
        for b in range(len(held_boxes)):
            far = torch.ones(pb.shape[1], dtype=torch.bool, device=pb.device)
            if len(all_boxes[b]):
                far = box_iou(pb[b], box_cxcywh_to_xyxy(all_boxes[b].float()))[0].max(-1).values < self.unkobj_neg_iou
            pos = torch.zeros_like(far)
            ignore = torch.zeros_like(far)
            if len(held_boxes[b]):
                ih = box_iou(pb[b], box_cxcywh_to_xyxy(held_boxes[b].float()))[0]          # [Q, Nh]
                near = ih.max(-1).values >= self.unkobj_pos_iou
                if self.unkobj_pos_top1:
                    best = ih.argmax(0)                                                     # best query per object
                    pos[best[ih.max(0).values >= self.unkobj_pos_iou]] = True
                    ignore = near & ~pos                                                    # other queries on it
                else:
                    pos = near
            neg = far
            if self.unkobj_dup_neg and known_boxes is not None and len(known_boxes[b]):
                ik = box_iou(pb[b], box_cxcywh_to_xyxy(known_boxes[b].float()))[0]          # [Q, Nk]
                dup = ik.max(-1).values >= 0.5
                dup[ik.argmax(0)] = False                                                   # keep each object's best query out
                neg = neg | (dup & ~pos & ~ignore)
            tgt[b] = pos.to(tgt.dtype); use[b] = pos | neg; n_pos += float(pos.sum())
        if not use.any():
            return logits.sum() * 0.
        loss = torchvision.ops.sigmoid_focal_loss(logits[use].float(), tgt[use], self.unk_alpha, self.gamma, reduction='sum')
        return loss / max(n_pos, 1.)

    def loss_ncobj(self, outputs, targets, indices, num_boxes, **kw):
        logits = outputs['pred_nc_logits']                        # [B, Q]
        tgt = torch.zeros_like(logits)
        tgt[self._get_src_permutation_idx(indices)] = 1.
        loss = torchvision.ops.sigmoid_focal_loss(logits.float(), tgt, 0.25, 2.0, reduction='sum') / num_boxes
        if self._epoch < self.attn_pseudo_start:
            loss = loss * 0.
        return {'loss_ncobj': loss}

    @torch.no_grad()
    def _unk_pseudo_score(self, outputs, heads, targets):
        """B: the model's own unknown score o * sigmoid(u) * sigmoid(q) (M0's learned_q), detached."""
        o = heads[outputs['obj_key']].objectness(outputs['pred_obj'].float())
        score = o * outputs['pred_unk_logits'].float().sigmoid() * outputs['pred_quality_logits'].float().sigmoid()
        if self.pseudo_known_iou > 0:
            xyxy = box_cxcywh_to_xyxy(outputs['pred_boxes'].detach().float())
            for b in range(len(score)):
                kb = box_cxcywh_to_xyxy(targets[b]['boxes'].float())
                if len(kb):
                    near = torchvision.ops.box_iou(xyxy[b], kb).max(1).values >= self.pseudo_known_iou
                    score[b][near] = -1e10
        return score

    @torch.no_grad()
    def _attn_pseudo(self, outputs, targets, indices, attn_map, heads=None):
        """OW-DETR: add top-k unmatched queries by in-box mean activation as unknown targets."""
        B, Q = outputs['pred_boxes'].shape[:2]
        if self.pseudo_score == 'unk':
            return self._select_pseudo(self._unk_pseudo_score(outputs, heads, targets), outputs, targets, indices)
        H = W = 640
        amap = F.interpolate(attn_map.unsqueeze(1), size=(H, W), mode='bilinear', align_corners=False)[:, 0]
        integ = F.pad(amap.cumsum(1).cumsum(2), (1, 0, 1, 0))                         # [B, H+1, W+1]
        xyxy = box_cxcywh_to_xyxy(outputs['pred_boxes'].detach().float())
        px = (xyxy * torch.tensor([W, H, W, H], device=xyxy.device)).long()
        x1, y1 = px[..., 0].clamp(0, W), px[..., 1].clamp(0, H)
        x2, y2 = px[..., 2].clamp(0, W), px[..., 3].clamp(0, H)
        bi = torch.arange(B, device=xyxy.device).unsqueeze(1).expand(B, Q)
        ssum = integ[bi, y2, x2] - integ[bi, y1, x2] - integ[bi, y2, x1] + integ[bi, y1, x1]
        area = ((x2 - x1) * (y2 - y1)).float()
        score = torch.where(area > 0, ssum / area.clamp(min=1), torch.full_like(ssum, -1e10))
        if self.cat_ss:
            score = self._cat_score(score, xyxy, targets, indices)
        return self._select_pseudo(score, outputs, targets, indices)

    @torch.no_grad()
    def _ext_pseudo(self, outputs, targets, indices):
        """G4-1: one unmatched query per offline pseudo box (highest IoU >= pseudo_ext_iou), greedy over boxes."""
        unk = self.num_classes - 1
        xyxy = box_cxcywh_to_xyxy(outputs['pred_boxes'].detach().float())
        new_t, new_i = [], []
        for b, (t, (src, tgt)) in enumerate(zip(targets, indices)):
            pb = self._ext_boxes[b]
            sel_q, sel_p = [], []
            if len(pb):
                ov = box_iou(xyxy[b], box_cxcywh_to_xyxy(pb.float()))[0]                  # [Q, P]
                ov[src.to(ov.device)] = -1.
                while True:
                    v, flat = ov.flatten().max(0)
                    if v < self.pseudo_ext_iou:
                        break
                    q, j = divmod(int(flat), ov.shape[1])
                    sel_q.append(q); sel_p.append(j)
                    ov[q] = -1.; ov[:, j] = -1.
            q = torch.as_tensor(sel_q, dtype=src.dtype, device=src.device)
            j = torch.as_tensor(sel_p, dtype=torch.long, device=t['boxes'].device)
            n = len(t['labels'])
            new_t.append({'labels': torch.cat([t['labels'], torch.full((len(j),), unk, dtype=t['labels'].dtype, device=t['labels'].device)]),
                          'boxes': torch.cat([t['boxes'], pb[j].to(t['boxes'].dtype)])})
            new_i.append((torch.cat([src, q]), torch.cat([tgt, torch.arange(n, n + len(j), dtype=tgt.dtype, device=tgt.device)])))
        return new_t, new_i

    def _select_pseudo(self, score, outputs, targets, indices):
        Q = outputs['pred_boxes'].shape[1]
        unk = self.num_classes - 1
        new_t, new_i = [], []
        for b, (t, (src, tgt)) in enumerate(zip(targets, indices)):
            s = score[b].clone()
            s[src] = -1e10
            if self.pseudo_excl_iou > 0:
                top = self._exclusive_topk(s, outputs['pred_boxes'][b].detach())
            else:
                top = s.topk(min(self.attn_pseudo_k, Q)).indices
            n = len(t['labels'])
            new_t.append({'labels': torch.cat([t['labels'], torch.full_like(top, unk)]),
                          'boxes': torch.cat([t['boxes'], outputs['pred_boxes'][b, top].detach()])})
            new_i.append((torch.cat([src, top.to(src.device)]),
                          torch.cat([tgt, torch.arange(n, n + len(top), device=tgt.device)])))
        return new_t, new_i

    def _exclusive_topk(self, s, boxes, n_cand=50):
        """Greedy top-k over the n_cand best queries, skipping any that overlaps an already picked one."""
        cand = s.topk(min(n_cand, len(s))).indices
        cand = cand[s[cand] > -1e9]
        ov = box_iou(box_cxcywh_to_xyxy(boxes[cand].float()), box_cxcywh_to_xyxy(boxes[cand].float()))[0] >= self.pseudo_excl_iou
        keep = []
        for j in range(len(cand)):
            if len(keep) == self.attn_pseudo_k:
                break
            if not any(ov[j, i] for i in keep):
                keep.append(j)
        return cand[torch.as_tensor(keep, dtype=torch.long, device=cand.device)]

    def _cat_score(self, s_o, xyxy, targets, indices):
        """CAT Eq. 3 on the unmatched queries of each image (matched / empty boxes keep -1e10)."""
        out = torch.full_like(s_o, -1e10)
        w_m, w_i = self.cat_w
        for b, (t, (src, _)) in enumerate(zip(targets, indices)):
            cand = s_o[b] > -1e9
            cand[src] = False
            if not cand.any():
                continue
            v = s_o[b][cand]
            nv = (v - v.min()) / (v.max() - v.min()).clamp(min=1e-12)
            ss = t.get('ss_boxes', None)
            if ss is None or len(ss) == 0:
                out[b][cand] = nv
                continue
            iou = box_iou(xyxy[b][cand], box_cxcywh_to_xyxy(ss.float()))[0].max(-1).values
            out[b][cand] = nv.clamp(min=0) ** w_m * iou.clamp(min=0) ** w_i
        return out

    @torch.no_grad()
    def _cat_update(self, total_loss):
        """CAT loss-memory measurer / sensor / adjuster (paper Eq. 6-8)."""
        self._loss_mem.append(float(total_loss))
        self._iter += 1
        if self._epoch < self.attn_pseudo_start or self._iter % self.cat_tb or len(self._loss_mem) < 40:
            return
        mem = list(self._loss_mem)
        w = torch.arange(20, 0, -1, dtype=torch.float64)            # newest entry weighted most
        w = w / w.sum()
        recent = (w * torch.tensor(mem[-1:-21:-1], dtype=torch.float64)).sum()
        older = (w * torch.tensor(mem[-21:-41:-1], dtype=torch.float64)).sum()
        dl = float(recent / older.clamp(min=1e-12))
        dw = 0.5 / (1 + math.exp(-(dl - 1))) if dl > 1 else -dl / 3
        w_m, w_i = self.cat_w[0] * (1 + dw), self.cat_w[1] * (1 - dw)
        tot = max(w_m + w_i, 1e-12)
        self.cat_w = [w_m / tot, w_i / tot]
        if self._iter % (self.cat_tb * 25) == 0:
            print(f'[CAT] iter {self._iter} dl {dl:.3f} W_m {self.cat_w[0]:.3f} W_I {self.cat_w[1]:.3f}', flush=True)

    def _randbox_unknown(self, outputs, best):
        """[B, Q] bool: top-k queries by sum_c sigmoid(logit_c) minus each GT's best query (RandBox loss.py:459-475)."""
        score = outputs['pred_logits'].detach().float().sigmoid().sum(-1)
        sel = torch.zeros_like(score, dtype=torch.bool)
        top = score.topk(min(self.randbox_unk_k, score.shape[1]), dim=1).indices
        sel.scatter_(1, top, True)
        for b, bq in enumerate(best):
            sel[b, bq] = False
        return sel

    def loss_randbox_unk(self, outputs, sel):
        logits = outputs['pred_logits']
        tgt = torch.zeros_like(logits)
        tgt[..., -1] = sel.to(tgt.dtype)
        loss = torchvision.ops.sigmoid_focal_loss(logits.float(), tgt, self.alpha, self.gamma, reduction='sum')
        loss = loss / max(float(sel.sum()), 1.)
        if self._train_iter < self.randbox_unk_start:
            loss = loss * 0.
        return loss

    def loss_decorr(self, outputs):
        """OrthogonalDet Eq. 4: mean over classes of the squared Pearson correlation between (detached) class
        probabilities and objectness over all queries of the batch."""
        P = outputs['pred_cls_prob'].detach().float().flatten(0, 1)                # [N, C]
        o = outputs['pred_pobj'].float().reshape(-1, 1)                           # [N, 1]
        Pc, oc = P - P.mean(0), o - o.mean(0)
        cov = (Pc * oc).sum(0)
        var = (Pc ** 2).sum(0) * (oc ** 2).sum(0)
        return (cov ** 2 / (var + 1e-8)).mean()

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'boxes': self.loss_boxes,
            'focal': self.loss_labels_focal,
            'vfl': self.loss_labels_vfl,
            'obj': self.loss_obj,
            'unk': self.loss_unk,
            'quality': self.loss_quality,
            'primary': self.loss_primary,
            'ncobj': self.loss_ncobj,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        if loss == 'obj' and 'pred_obj' not in outputs:
            return {}
        if loss == 'unk' and 'pred_unk_logits' not in outputs:
            return {}
        if loss == 'quality' and 'pred_quality_logits' not in outputs:
            return {}
        if loss == 'primary' and 'pred_primary_logits' not in outputs:
            return {}
        if loss == 'ncobj' and 'pred_nc_logits' not in outputs:
            return {}
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    # ------------------------------------------------------------------ C3
    def _bg_ignore_mask(self, outputs, indices):
        if self.bg_ignore_topk <= 0 or 'pred_obj' not in outputs:
            return None
        d2 = outputs['pred_obj'].detach()
        matched = torch.zeros_like(d2, dtype=torch.bool)
        matched[self._get_src_permutation_idx(indices)] = True
        score = (-d2).masked_fill(matched, float('-inf'))
        k = min(self.bg_ignore_topk, d2.shape[1])
        top = score.topk(k, dim=1).indices
        ignore = torch.zeros_like(matched)
        ignore.scatter_(1, top, True)
        return ignore & ~matched

    def _e_step(self, outputs, indices, heads):
        if heads is None or 'pred_embed' not in outputs:
            return
        key = outputs['obj_key']
        if self.obj_stats == 'all' and key.startswith('dec'):
            heads[key].accumulate(outputs['pred_embed'].flatten(0, 1))
            return
        idx = self._get_src_permutation_idx(indices)
        heads[key].accumulate(outputs['pred_embed'][idx])

    def _apply_losses(self, outputs, targets, indices, num_boxes, suffix, ignore=None, cls_ti=None):
        """cls_ti = (targets, indices) incl. pseudo-unknowns, used by the classification / objectness losses only."""
        losses = {}
        for loss in self.losses:
            cls_loss = loss in ('vfl', 'focal', 'ncobj') or (loss == 'primary' and self.pseudo_primary)
            t, i = cls_ti if (cls_ti is not None and cls_loss) else (targets, indices)
            meta = self.get_loss_meta_info(loss, outputs, t, i)
            if loss in ('vfl', 'focal'):
                meta['ignore'] = ignore
            if loss == 'unk' and cls_ti is not None:
                meta['pseudo_ti'] = cls_ti
            l_dict = self.get_loss(loss, outputs, t, i, num_boxes, **meta)
            l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
            losses.update({k + suffix: v for k, v in l_dict.items()})
        return losses

    # ------------------------------------------------------------------ forward
    def forward(self, outputs, targets, **kwargs):
        heldout = None
        if any('heldout_mask' in t for t in targets):
            all_boxes = [t['boxes'] for t in targets]
            held_boxes = [t['boxes'][t['heldout_mask']] for t in targets]
            targets = [{'labels': t['labels'][~t['heldout_mask']], 'boxes': t['boxes'][~t['heldout_mask']]}
                       for t in targets]
            heldout = (held_boxes, all_boxes, [t['boxes'] for t in targets])
        if self.pseudo_score == 'ext':
            unk = self.num_classes - 1
            self._ext_boxes = [t['boxes'][t['labels'] == unk] for t in targets]
            targets = [{'labels': t['labels'][t['labels'] != unk], 'boxes': t['boxes'][t['labels'] != unk]} for t in targets]
        losses = self._forward(outputs, targets, **kwargs)
        if self.cat_ss and self.training:
            self._cat_update(sum(v.detach() for v in losses.values()))
        if heldout is not None and 'loss_unkobj' in self.weight_dict:
            w = self.weight_dict['loss_unkobj']
            for i, o in enumerate([outputs] + list(outputs.get('aux_outputs', []))):
                if 'pred_unkobj_logits' in o:
                    losses['loss_unkobj' + ('' if i == 0 else f'_aux_{i - 1}')] = w * self._loss_unkobj(o, *heldout)
        return losses

    def _forward_randbox(self, outputs, targets):
        """RandBox / OrthogonalDet: SimOTA matching per layer, known losses normalised by matched-query count,
        unknown loss on the top-k matching-score queries, optional decorrelation loss. No DN / encoder outputs."""
        if self.training:
            self._train_iter += 1
        losses = {}
        layers = [(outputs, '')] + [(a, f'_aux_{i}') for i, a in enumerate(outputs.get('aux_outputs', []))]
        for o, suffix in layers:
            m = self.matcher(o, targets)
            indices = m['indices']
            nb = max(float(sum(len(s) for s, _ in indices)), 1.)
            losses.update(self._apply_losses(o, targets, indices, nb, suffix))
            if self.randbox_unk_k > 0 and 'loss_randbox_unk' in self.weight_dict:
                sel = self._randbox_unknown(o, m['best'])
                losses['loss_randbox_unk' + suffix] = self.weight_dict['loss_randbox_unk'] * self.loss_randbox_unk(o, sel)
            if self.decorr and 'loss_decorr' in self.weight_dict:
                losses['loss_decorr' + suffix] = self.weight_dict['loss_decorr'] * self.loss_decorr(o)
        return losses

    def _o2m_losses(self, layers, targets, num_boxes):
        K = self.o2m_k
        rep_t = [{'labels': t['labels'].repeat(K), 'boxes': t['boxes'].repeat(K, 1)} for t in targets]
        orig = self.num_classes
        if self.o2m_known_slots is not None:
            self.num_classes = self.o2m_known_slots
        losses = {}
        try:
            for i, o in enumerate(layers):
                if self.o2m_known_slots is not None:
                    o = {'pred_logits': o['pred_logits'][..., :self.o2m_known_slots], 'pred_boxes': o['pred_boxes']}
                indices = self.matcher(o, rep_t)['indices']
                for k, v in self._apply_losses(o, rep_t, indices, num_boxes * K, f'_o2m_{i}').items():
                    losses[k] = v * self.o2m_weight
        finally:
            self.num_classes = orig
        return losses

    def _pseudo(self, outputs, targets, indices, attn_map, heads=None):
        if self.pseudo_score == 'ext':
            if self._ext_boxes is None or self._epoch < self.attn_pseudo_start:
                return None
            return self._ext_pseudo(outputs, targets, indices)
        if self.attn_pseudo_k <= 0 or self._epoch < self.attn_pseudo_start:
            return None
        if self.pseudo_score == 'attn' and attn_map is None:
            return None
        return self._attn_pseudo(outputs, targets, indices, attn_map, heads)

    def _forward(self, outputs, targets, **kwargs):
        self._epoch = kwargs.get('epoch', self._epoch)
        attn_map = outputs.get('attn_map', None)
        heads = outputs.get('obj_heads', None)
        self._obj_heads = heads
        outputs_without_aux = {k: v for k, v in outputs.items() if 'aux' not in k}

        if self.randbox_unk_k > 0 or self.decorr:
            return self._forward_randbox(outputs, targets)

        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=outputs['pred_logits'].device)
        if is_dist_available_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        indices = self.matcher(outputs_without_aux, targets)['indices']
        self._e_step(outputs_without_aux, indices, heads)
        losses = self._apply_losses(outputs_without_aux, targets, indices, num_boxes, '',
                                    self._bg_ignore_mask(outputs_without_aux, indices),
                                    self._pseudo(outputs_without_aux, targets, indices, attn_map, heads))

        for i, aux in enumerate(outputs.get('aux_outputs', [])):
            if not self.share_matched_indices:
                indices = self.matcher(aux, targets)['indices']
            self._e_step(aux, indices, heads)
            losses.update(self._apply_losses(aux, targets, indices, num_boxes, f'_aux_{i}',
                                             self._bg_ignore_mask(aux, indices),
                                             self._pseudo(aux, targets, indices, attn_map, heads)))

        if 'dn_aux_outputs' in outputs:
            dn_meta = outputs['dn_meta']
            dn_targets = dn_meta.get('dn_targets', targets)
            indices = self.get_cdn_matched_indices(dn_meta, dn_targets)
            dn_num_boxes = num_boxes * dn_meta['dn_num_group']
            for i, aux in enumerate(outputs['dn_aux_outputs']):
                losses.update(self._apply_losses(aux, dn_targets, indices, dn_num_boxes, f'_dn_{i}'))

        if self.o2m_k > 0 and 'o2m_outputs' in outputs:
            losses.update(self._o2m_losses(outputs['o2m_outputs'], targets, num_boxes))

        if 'enc_aux_outputs' in outputs:
            class_agnostic = outputs['enc_meta']['class_agnostic']
            if class_agnostic:
                orig_num_classes = self.num_classes
                self.num_classes = 1
                enc_targets = copy.deepcopy(targets)
                for t in enc_targets:
                    t['labels'] = torch.zeros_like(t["labels"])
            else:
                enc_targets = targets
            for i, aux in enumerate(outputs['enc_aux_outputs']):
                indices = self.matcher(aux, targets)['indices']
                self._e_step(aux, indices, heads)
                ignore = self._bg_ignore_mask(aux, indices) if self.bg_ignore_enc else None
                losses.update(self._apply_losses(aux, enc_targets, indices, num_boxes, f'_enc_{i}', ignore))
            if class_agnostic:
                self.num_classes = orig_num_classes

        return losses
