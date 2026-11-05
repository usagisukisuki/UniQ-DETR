"""SimOTA dynamic-k matcher of RandBox / DiffusionDet (third_party/baselines_src/RandBox/randbox/loss.py:381-578).

One-to-many: each GT g takes its k_g = max(1, int(sum of its top-5 IoUs)) lowest-cost predictions; a prediction matched
to several GTs keeps its argmin-cost GT. cost = 5*L1(normalised xyxy) + 2*focal_cls + 2*(-GIoU)
+ 100*(not in box AND in centre) + 1e4*(not in box OR centre), centre radius 2.5 x GT size.
Returns {'indices': [(src, tgt)], 'best': [best query per GT]} (the latter is RandBox's matched_qidx).
The stale `anchor_matching_gt` reuse of the original (loss.py:565) is fixed here.
"""
import torch
import torch.nn as nn

from src.core import register
from src.zoo.rtdetr.box_ops import box_cxcywh_to_xyxy, box_iou, generalized_box_iou

__all__ = ['SimOTAMatcher']


@register()
class SimOTAMatcher(nn.Module):
    def __init__(self, cost_class=2.0, cost_bbox=5.0, cost_giou=2.0, alpha=0.25, gamma=2.0, ota_k=5, center_radius=2.5,
                 use_focal_loss=True, weight_dict=None):
        super().__init__()
        if weight_dict:   # inherited HungarianMatcher-style config
            cost_class = weight_dict.get('cost_class', cost_class)
            cost_bbox = weight_dict.get('cost_bbox', cost_bbox)
            cost_giou = weight_dict.get('cost_giou', cost_giou)
        self.cost_class, self.cost_bbox, self.cost_giou = cost_class, cost_bbox, cost_giou
        self.alpha, self.gamma, self.ota_k, self.center_radius = alpha, gamma, ota_k, center_radius

    def _in_boxes(self, p, g):
        """p [Q,4], g [G,4] cxcywh -> (fg_mask [Q], in_box_and_center [Q,G])"""
        gx = box_cxcywh_to_xyxy(g)
        cx, cy = p[:, 0:1], p[:, 1:2]
        in_box = (cx > gx[:, 0]) & (cx < gx[:, 2]) & (cy > gx[:, 1]) & (cy < gx[:, 3])
        rw, rh = self.center_radius * (gx[:, 2] - gx[:, 0]), self.center_radius * (gx[:, 3] - gx[:, 1])
        in_ctr = (cx > g[:, 0] - rw) & (cx < g[:, 0] + rw) & (cy > g[:, 1] - rh) & (cy < g[:, 1] + rh)
        return in_box.any(1) | in_ctr.any(1), in_box & in_ctr

    def _dynamic_k(self, cost, ious):
        Q, G = cost.shape
        mm = torch.zeros_like(cost)
        ks = torch.clamp(ious.topk(min(self.ota_k, Q), dim=0).values.sum(0).int(), min=1)
        for g in range(G):
            mm[cost[:, g].topk(int(ks[g]), largest=False).indices, g] = 1.
        multi = mm.sum(1) > 1
        if multi.any():
            am = cost[multi].argmin(1)
            mm[multi] = 0
            mm[multi, am] = 1
        while (mm.sum(0) == 0).any():
            cost = cost + (mm.sum(1, keepdim=True) > 0) * 1e5
            for g in torch.nonzero(mm.sum(0) == 0).squeeze(1):
                mm[cost[:, g].argmin(), g] = 1.
            multi = mm.sum(1) > 1
            if multi.any():
                am = cost[multi].argmin(1)
                mm[multi] = 0
                mm[multi, am] = 1
        sel = mm.sum(1) > 0
        src = torch.nonzero(sel).squeeze(1)
        tgt = mm[sel].argmax(1)
        best = cost.masked_fill(mm == 0, float('inf')).argmin(0)
        return src, tgt, best

    @torch.no_grad()
    def forward(self, outputs, targets):
        prob = outputs['pred_logits'].float().sigmoid()
        boxes = outputs['pred_boxes'].float()
        indices, best = [], []
        for b, t in enumerate(targets):
            dev = boxes.device
            if len(t['labels']) == 0:
                e = torch.zeros(0, dtype=torch.int64, device=dev)
                indices.append((e, e)); best.append(e)
                continue
            p, g = boxes[b], t['boxes'].float()
            pr = prob[b][:, t['labels']]
            a, gm = self.alpha, self.gamma
            cost_cls = a * (1 - pr) ** gm * (-(pr + 1e-8).log()) - (1 - a) * pr ** gm * (-(1 - pr + 1e-8).log())
            px, gx = box_cxcywh_to_xyxy(p), box_cxcywh_to_xyxy(g)
            fg, inbc = self._in_boxes(p, g)
            cost = (self.cost_bbox * torch.cdist(px, gx, p=1) + self.cost_class * cost_cls
                    - self.cost_giou * generalized_box_iou(px, gx) + 100. * (~inbc))
            cost[~fg] += 1e4
            src, tgt, bq = self._dynamic_k(cost, box_iou(px, gx)[0])
            indices.append((src, tgt)); best.append(bq)
        return {'indices': indices, 'best': best}
