"""Open-world RT-DETRv2 decoder.

C1: class-agnostic objectness on the decoder query embedding (separate from the cls head).
C2: objectness = Gaussian (Mahalanobis) likelihood of the query embedding, one Gaussian per
    decoder layer (+ one for the encoder when C4 is on). See gaussian_objectness.py.
C4: `query_select_method='objectness'` selects the top-K encoder tokens by Gaussian
    objectness instead of max_c sigmoid(cls_logit_c) (RT-DETR default).

Extra output keys (consumed by OWCriterion / OWPostProcessor):
  pred_obj      [B, Q]  squared Mahalanobis distance d2 of the last layer (smaller = more object-like)
  pred_embed    [B, Q, D] last-layer query embedding (training only, for E-step)
  aux_outputs[i] / enc_aux_outputs[0] carry the same keys for their layer.
  obj_heads     dict name -> GaussianObjectness (training only, for E-step accumulation)
"""

from typing import List

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core import register
from src.zoo.rtdetr.rtdetrv2_decoder import RTDETRTransformerv2
from src.zoo.rtdetr.denoising import get_contrastive_denoising_training_group
from src.zoo.rtdetr.utils import inverse_sigmoid

from .gaussian_objectness import GaussianObjectness

__all__ = ['OWRTDETRTransformerv2']


@register()
class OWRTDETRTransformerv2(RTDETRTransformerv2):
    __share__ = ['num_classes', 'eval_spatial_size']

    def __init__(self,
                 num_classes=80,
                 hidden_dim=256,
                 num_queries=300,
                 feat_channels=[512, 1024, 2048],
                 feat_strides=[8, 16, 32],
                 num_levels=3,
                 num_points=4,
                 nhead=8,
                 num_layers=6,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation="relu",
                 num_denoising=100,
                 label_noise_ratio=0.5,
                 box_noise_scale=1.0,
                 learn_query_content=False,
                 eval_spatial_size=None,
                 eval_idx=-1,
                 eps=1e-2,
                 aux_loss=True,
                 cross_attn_method='default',
                 query_select_method='default',
                 # --- open-world ---
                 use_objectness=False,         # C1 + C2
                 obj_cov_type='full',          # full | diag
                 obj_momentum=0.1,
                 obj_shrinkage=1e-3,
                 obj_interval=50,
                 obj_temperature=1.3,
                 use_unk_logit=False,          # C1 (PROB-style): learned "not a known object" logit
                 use_quality=False,            # F1: class-agnostic localisation-quality (IoU) head, OLN-style
                 use_primary=False,            # G: class-agnostic one-to-one ('primary query of its object') head
                 use_unkobj=False,             # F3: 'unlabelled object' head trained on held-out known classes
                 heldout_k=0,                  # F3: known classes treated as unknown per training iteration
                 pseudo_label=None,            # G4-1: label of offline pseudo-unknown boxes in the targets (kept out of DN)
                 o2m_queries=0,                # G4-3: training-only one-to-many query group (encoder top-N), 0 = off
                 use_nc_obj=False,             # OW-DETR / CAT: binary foreground-objectness branch (Linear 256 -> 1)
                 cat_cascade=False,            # CAT: localisation pass then identification pass (second decoder)
                 random_queries=False,         # RandBox: random boxes replace encoder query selection (no DN)
                 random_train_n=500,           # RandBox: random boxes per image in training
                 random_chunk=500,             # RandBox: inference grid processed in chunks of this many boxes
                 random_grid=10,               # RandBox: grid values per dimension (cx, cy, w, h) at inference
                 random_min_wh=0.02,           # RandBox: lower bound on w, h (zero-size boxes break inverse_sigmoid)
                 polar_head=False,             # OrthogonalDet: polar (radius = objectness, angle = class) head
                 polar_slots=80,               # OrthogonalDet: known class slots (80 as in the original; 20 used)
                 ):
        assert query_select_method in ('default', 'one2many', 'agnostic', 'objectness')
        super().__init__(
            num_classes=num_classes, hidden_dim=hidden_dim, num_queries=num_queries,
            feat_channels=feat_channels, feat_strides=feat_strides, num_levels=num_levels,
            num_points=num_points, nhead=nhead, num_layers=num_layers, dim_feedforward=dim_feedforward,
            dropout=dropout, activation=activation, num_denoising=num_denoising,
            label_noise_ratio=label_noise_ratio, box_noise_scale=box_noise_scale,
            learn_query_content=learn_query_content, eval_spatial_size=eval_spatial_size,
            eval_idx=eval_idx, eps=eps, aux_loss=aux_loss, cross_attn_method=cross_attn_method,
            query_select_method='default' if query_select_method == 'objectness' else query_select_method)
        self.query_select_method = query_select_method
        self.use_objectness = use_objectness
        if query_select_method == 'objectness':
            assert use_objectness, 'C4 (objectness query selection) requires C1/C2'

        gkw = dict(dim=hidden_dim, momentum=obj_momentum, shrinkage=obj_shrinkage, interval=obj_interval,
                   temperature=obj_temperature, cov_type=obj_cov_type)
        if use_objectness:
            self.dec_obj_head = nn.ModuleList([GaussianObjectness(**gkw) for _ in range(num_layers)])
        if query_select_method == 'objectness':
            self.enc_obj_head = GaussianObjectness(**gkw)
        self.random_queries = random_queries
        if random_queries:
            # RandBox (docs/baselines/RANDBOX_SPEC.md): proposals independent of the data. Content query = mean encoder
            # feature inside the box (RoI mean, as RandBox's first-stage proposal feature) -> Linear + LN.
            self.random_train_n, self.random_chunk, self.random_min_wh = random_train_n, random_chunk, random_min_wh
            self.rand_content = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim))
            v = (torch.arange(random_grid, dtype=torch.float32) + 0.5) / random_grid           # centres 0.05..0.95
            s_ = torch.arange(1, random_grid + 1, dtype=torch.float32) / random_grid           # sizes 0.1..1.0
            grid = torch.stack(torch.meshgrid(v, v, s_, s_, indexing='ij'), -1).reshape(-1, 4)
            g = torch.Generator().manual_seed(0)
            self.register_buffer('rand_grid', grid[torch.randperm(len(grid), generator=g)], persistent=False)
        self.polar_head = polar_head
        if polar_head:
            # OrthogonalDet (docs/baselines/ORTHOGONALDET_SPEC.md): f = LN(Linear_nobias(q)); p_obj = sigmoid(BN1d(|f|));
            # logits = Linear(LN(f)) over polar_slots known slots + raw unknown; unknown synthesised by Eq. 3;
            # p_joint = softmax * p_obj. pred_logits = logit(p_joint) so losses / matcher / postprocess use p_joint.
            self.polar_slots = polar_slots
            self.polar_f = nn.ModuleList([nn.Sequential(nn.Linear(hidden_dim, hidden_dim, bias=False), nn.LayerNorm(hidden_dim))
                                          for _ in range(num_layers)])
            self.polar_dir = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
            self.polar_obj = nn.ModuleList([nn.BatchNorm1d(1) for _ in range(num_layers)])
            self.polar_cls = nn.ModuleList([nn.Linear(hidden_dim, polar_slots + 1) for _ in range(num_layers)])
            for h in self.polar_cls:
                nn.init.xavier_uniform_(h.weight)
                nn.init.constant_(h.bias, -math.log((1 - 0.01) / 0.01))
        self.cat_cascade = cat_cascade
        if cat_cascade:
            # CAT (docs/baselines/CAT_SPEC.md): the identification pass runs a decoder with its own weights (as in the
            # SKDF code; the paper shares F_s), with query_pos = last localisation-pass embedding, content = 0 and the
            # localisation pass's final boxes as (fixed) reference. Class logits come from this pass only.
            self.class_decoder_layers = copy.deepcopy(self.decoder.layers)
            self.dec2_score_head = nn.ModuleList([nn.Linear(hidden_dim, num_classes) for _ in range(num_layers)])
            for h in self.dec2_score_head:
                nn.init.constant_(h.bias, -math.log((1 - 0.01) / 0.01))
        self.use_nc_obj = use_nc_obj
        if use_nc_obj:
            self.dec_nc_head = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(num_layers)])
            for h in self.dec_nc_head:
                nn.init.constant_(h.bias, -math.log((1 - 0.01) / 0.01))
        self.use_unkobj = use_unkobj
        self.heldout_k = heldout_k
        self.pseudo_label = pseudo_label
        self.o2m_queries = o2m_queries
        if use_unkobj:
            self.dec_unkobj_head = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(num_layers)])
            for h in self.dec_unkobj_head:
                nn.init.constant_(h.bias, -math.log((1 - 0.01) / 0.01))
        self.use_quality = use_quality
        if use_quality:
            self.dec_quality_head = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(num_layers)])
        self.use_primary = use_primary
        if use_primary:
            self.dec_primary_head = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(num_layers)])
        self.use_unk_logit = use_unk_logit
        if use_unk_logit:
            # one logit per decoder layer, trained with target 1 on unmatched queries (PROB's class 80)
            self.dec_unk_head = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(num_layers)])
            bias = -math.log((1 - 0.01) / 0.01)
            for h in self.dec_unk_head:
                nn.init.constant_(h.bias, bias)

    # ------------------------------------------------------------------ encoder -> queries
    def _get_decoder_input(self, memory, spatial_shapes, denoising_logits=None, denoising_bbox_unact=None):
        if self.training or self.eval_spatial_size is None:
            anchors, valid_mask = self._generate_anchors(spatial_shapes, device=memory.device)
        else:
            anchors, valid_mask = self.anchors, self.valid_mask

        memory = valid_mask.to(memory.dtype) * memory
        output_memory = self.enc_output(memory)
        enc_outputs_logits = self.enc_score_head(output_memory)
        enc_outputs_coord_unact = self.enc_bbox_head(output_memory) + anchors

        if self.query_select_method == 'objectness':
            enc_d2 = self.enc_obj_head.mahalanobis(output_memory)            # [B, L]
            # invalid anchors (outside image) must never be selected
            enc_d2 = enc_d2.masked_fill(~valid_mask.squeeze(-1).expand_as(enc_d2), float('inf'))
            _, topk_ind = torch.topk(-enc_d2, self.num_queries, dim=-1)
        else:
            enc_d2 = None
            topk_ind = self._topk_index(enc_outputs_logits)

        gather = lambda t: t.gather(1, topk_ind.unsqueeze(-1).repeat(1, 1, t.shape[-1]))
        enc_topk_memory = gather(output_memory)
        enc_topk_logits = gather(enc_outputs_logits)
        enc_topk_bbox_unact = gather(enc_outputs_coord_unact)

        enc_aux = []
        if self.training:
            aux = {'pred_logits': enc_topk_logits, 'pred_boxes': F.sigmoid(enc_topk_bbox_unact)}
            if enc_d2 is not None:
                aux['pred_obj'] = enc_d2.gather(1, topk_ind)
                aux['pred_embed'] = enc_topk_memory
                aux['obj_key'] = 'enc'
            enc_aux.append(aux)

        if self.learn_query_content:
            content = self.tgt_embed.weight.unsqueeze(0).tile([memory.shape[0], 1, 1])
        else:
            content = enc_topk_memory.detach()
        enc_topk_bbox_unact = enc_topk_bbox_unact.detach()

        if denoising_bbox_unact is not None:
            enc_topk_bbox_unact = torch.concat([denoising_bbox_unact, enc_topk_bbox_unact], dim=1)
            content = torch.concat([denoising_logits, content], dim=1)

        if self.training and self.o2m_queries > 0:
            # G4-3 (H-DETR hybrid branch): a separate group of the encoder's top-o2m_queries proposals, appended last
            ind = torch.topk(enc_outputs_logits.max(-1).values, self.o2m_queries, dim=-1)[1].unsqueeze(-1)
            content = torch.concat([content, output_memory.gather(1, ind.repeat(1, 1, output_memory.shape[-1])).detach()], dim=1)
            enc_topk_bbox_unact = torch.concat([enc_topk_bbox_unact, enc_outputs_coord_unact.gather(1, ind.repeat(1, 1, 4)).detach()], dim=1)

        return content, enc_topk_bbox_unact, enc_aux

    def _topk_index(self, logits):
        if self.query_select_method == 'default':
            return torch.topk(logits.max(-1).values, self.num_queries, dim=-1)[1]
        if self.query_select_method == 'one2many':
            return torch.topk(logits.flatten(1), self.num_queries, dim=-1)[1] // self.num_classes
        if self.query_select_method == 'agnostic':
            return torch.topk(logits.squeeze(-1), self.num_queries, dim=-1)[1]
        raise ValueError(self.query_select_method)

    # ------------------------------------------------------------------ decoder
    def _run_decoder(self, target, ref_points_unact, memory, spatial_shapes, attn_mask):
        """Same as TransformerDecoder.forward but also returns per-layer query embeddings."""
        dec = self.decoder
        bboxes, logits, embeds = [], [], []
        ref_points_detach = F.sigmoid(ref_points_unact)
        output = target
        for i, layer in enumerate(dec.layers):
            ref_points_input = ref_points_detach.unsqueeze(2)
            query_pos_embed = self.query_pos_head(ref_points_detach)
            output = layer(output, ref_points_input, memory, spatial_shapes, attn_mask, None, query_pos_embed)
            inter_ref_bbox = F.sigmoid(self.dec_bbox_head[i](output) + inverse_sigmoid(ref_points_detach))

            if self.training:
                logits.append(self.dec_score_head[i](output))
                embeds.append(output)
                if i == 0:
                    bboxes.append(inter_ref_bbox)
                else:
                    bboxes.append(F.sigmoid(self.dec_bbox_head[i](output) + inverse_sigmoid(ref_points)))
            elif i == dec.eval_idx:
                logits.append(self.dec_score_head[i](output))
                embeds.append(output)
                bboxes.append(inter_ref_bbox)
                break

            ref_points = inter_ref_bbox
            ref_points_detach = inter_ref_bbox.detach()
        return torch.stack(bboxes), torch.stack(logits), torch.stack(embeds)

    def _polar(self, layer, q):
        """OrthogonalDet head on query embeddings q [B, Q, D] -> (logit(p_joint) [B,Q,K+1], softmax probs, p_obj)."""
        f = self.polar_f[layer](q)
        r = f.norm(dim=-1, keepdim=True)
        p_obj = torch.sigmoid(self.polar_obj[layer](r.flatten(0, 1)).view_as(r)).squeeze(-1)       # [B, Q]
        l = self.polar_cls[layer](self.polar_dir[layer](f)).float()
        K = self.polar_slots
        z = l - l[..., :K].max(-1, keepdim=True).values
        lse_all = torch.logsumexp(z[..., :K], -1)                                                  # log S
        # log(S - 1) = logsumexp over the non-max known logits (S = 1 + sum of the others since max z = 0)
        zk = z[..., :K].clone()
        zk.scatter_(-1, zk.argmax(-1, keepdim=True), float('-inf'))
        lse_rest = torch.logsumexp(zk, -1)
        l_unk = lse_rest + lse_all - F.softplus(z[..., K])
        logits = torch.cat([z[..., :K], l_unk.unsqueeze(-1)], -1)
        prob = logits.softmax(-1)
        pj = (prob * p_obj.float().unsqueeze(-1)).clamp(1e-6, 1 - 1e-6)
        return torch.log(pj) - torch.log1p(-pj), prob, p_obj

    def _box_content(self, memory, spatial_shapes, boxes, n_pts=7):
        """Mean of the (enc_output-projected) encoder memory inside each box, averaged over levels. boxes [B,N,4] cxcywh."""
        mem = self.enc_output(memory)
        B, N, _ = boxes.shape
        t = (torch.arange(n_pts, device=boxes.device, dtype=boxes.dtype) + 0.5) / n_pts - 0.5          # [-0.5, 0.5)
        gy, gx = torch.meshgrid(t, t, indexing='ij')
        off = torch.stack([gx, gy], -1).reshape(1, 1, -1, 2)                                           # [1,1,P,2]
        pts = boxes[..., None, :2] + off * boxes[..., None, 2:]                                       # [B,N,P,2] in [0,1]
        grid = pts * 2 - 1
        out, start = 0., 0
        for (h, w) in spatial_shapes:
            fm = mem[:, start:start + h * w].transpose(1, 2).reshape(B, -1, h, w)
            out = out + F.grid_sample(fm, grid, align_corners=False).mean(-1).transpose(1, 2)          # [B,N,D]
            start += h * w
        return self.rand_content(out / len(spatial_shapes))

    def _random_boxes(self, B, device):
        z = torch.randn(B, self.random_train_n, 4, device=device).clamp(-2., 2.)
        b = (z + 2.) / 4.
        b[..., 2:] = b[..., 2:].clamp(min=self.random_min_wh)
        return b

    def _run_identification(self, loc_embed, loc_boxes, memory, spatial_shapes, attn_mask):
        """CAT identification pass (Eq. 2: E_class = F_s(M, E_location, R)). Returns class logits [L, B, Q, C]
        (training: all layers, paired with localisation layer i; eval: last layer only)."""
        ref = loc_boxes.detach().unsqueeze(2)
        output = torch.zeros_like(loc_embed)
        logits = []
        for i, layer in enumerate(self.class_decoder_layers):
            output = layer(output, ref, memory, spatial_shapes, attn_mask, None, loc_embed)
            if self.training or i == len(self.class_decoder_layers) - 1:
                logits.append(self.dec2_score_head[i](output))
        return torch.stack(logits)

    def _decode_boxes(self, boxes, memory, spatial_shapes):
        content = self._box_content(memory, spatial_shapes, boxes)
        return self._run_decoder(content, inverse_sigmoid(boxes), memory, spatial_shapes, None)

    def _layer_out(self, layer, logits, boxes, embeds):
        o = {'pred_logits': logits, 'pred_boxes': boxes}
        if self.polar_head:
            o['pred_logits'], o['pred_cls_prob'], o['pred_pobj'] = self._polar(layer, embeds)
        return o

    def _forward_random(self, memory, spatial_shapes):
        """RandBox query generation: training = fresh random boxes; eval = fixed grid in chunks (concatenated)."""
        B = memory.shape[0]
        if self.training:
            out_bboxes, out_logits, out_embeds = self._decode_boxes(self._random_boxes(B, memory.device), memory, spatial_shapes)
            outs = [self._layer_out(j, out_logits[j], out_bboxes[j], out_embeds[j]) for j in range(out_logits.shape[0])]
            out = outs[-1]
            out['aux_outputs'] = outs[:-1]
            return out
        chunks = []
        for c in self.rand_grid.split(self.random_chunk):
            bb, ll, ee = self._decode_boxes(c.unsqueeze(0).expand(B, -1, -1).contiguous(), memory, spatial_shapes)
            chunks.append(self._layer_out(self.decoder.eval_idx, ll[-1], bb[-1], ee[-1]))
        return {k: torch.cat([c[k] for c in chunks], 1) for k in chunks[0]}

    def forward(self, feats: List[torch.Tensor], targets=None):
        memory, spatial_shapes = self._get_encoder_input(feats)
        if self.random_queries:
            return self._forward_random(memory, spatial_shapes)

        if self.training and self.heldout_k > 0 and targets is not None:
            # F3: simulate unknowns with k random known classes. They are hidden from denoising here
            # (C6 cdn_mask) and from matching / all known-object losses in OWCriterion (heldout_mask);
            # only the 'unlabelled object' head sees them, as positives.
            held = torch.randperm(self.num_classes, device=targets[0]['labels'].device)[:self.heldout_k]
            for t in targets:
                m = torch.isin(t['labels'], held)
                t['heldout_mask'] = m
                t['cdn_mask'] = (t['cdn_mask'] & ~m) if 'cdn_mask' in t else ~m

        if self.training and self.pseudo_label is not None and targets is not None:
            # G4-1: offline pseudo-unknown boxes are not GT; OWCriterion splits them off, DN must not see them
            for t in targets:
                m = t['labels'] != self.pseudo_label
                t['cdn_mask'] = (t['cdn_mask'] & m) if 'cdn_mask' in t else m

        if self.training and self.num_denoising > 0:
            dn_targets = targets
            if targets is not None and any('cdn_mask' in t for t in targets):
                # C6: pseudo-unknown GT is excluded from contrastive denoising
                dn_targets = [{'labels': t['labels'][t['cdn_mask']], 'boxes': t['boxes'][t['cdn_mask']]}
                              if 'cdn_mask' in t else t for t in targets]
            denoising_logits, denoising_bbox_unact, attn_mask, dn_meta = \
                get_contrastive_denoising_training_group(
                    dn_targets, self.num_classes, self.num_queries, self.denoising_class_embed,
                    num_denoising=self.num_denoising, label_noise_ratio=self.label_noise_ratio,
                    box_noise_scale=self.box_noise_scale)
            if dn_meta is not None:
                dn_meta['dn_targets'] = dn_targets
        else:
            denoising_logits, denoising_bbox_unact, attn_mask, dn_meta = None, None, None, None

        init_ref_contents, init_ref_points_unact, enc_aux = \
            self._get_decoder_input(memory, spatial_shapes, denoising_logits, denoising_bbox_unact)

        o2m = self.training and self.o2m_queries > 0
        if o2m:
            # the one-to-many group and the rest (DN + one-to-one) cannot see each other
            T, M = init_ref_contents.shape[1], self.o2m_queries
            mask = torch.zeros(T, T, dtype=torch.bool, device=init_ref_contents.device)
            if attn_mask is not None:
                mask[:T - M, :T - M] = attn_mask
            mask[:T - M, T - M:] = True
            mask[T - M:, :T - M] = True
            attn_mask = mask

        out_bboxes, out_logits, out_embeds = self._run_decoder(
            init_ref_contents, init_ref_points_unact, memory, spatial_shapes, attn_mask)
        if o2m:
            out_bboxes, o2m_bboxes = out_bboxes[:, :, :-M], out_bboxes[:, :, -M:]
            out_logits, o2m_logits = out_logits[:, :, :-M], out_logits[:, :, -M:]
            out_embeds = out_embeds[:, :, :-M]
        if self.cat_cascade:
            out_logits = self._run_identification(out_embeds[-1], out_bboxes[-1], memory, spatial_shapes, attn_mask)

        if self.training and dn_meta is not None:
            split = dn_meta['dn_num_split']
            dn_out_bboxes, out_bboxes = torch.split(out_bboxes, split, dim=2)
            dn_out_logits, out_logits = torch.split(out_logits, split, dim=2)
            _, out_embeds = torch.split(out_embeds, split, dim=2)

        n_out = out_logits.shape[0]
        # layer index of each stacked output (eval: only eval_idx is computed)
        layer_ids = list(range(n_out)) if self.training else [self.decoder.eval_idx]

        outs = []
        for j in range(n_out):
            o = {'pred_logits': out_logits[j], 'pred_boxes': out_bboxes[j]}
            if self.use_unk_logit:
                o['pred_unk_logits'] = self.dec_unk_head[layer_ids[j]](out_embeds[j]).squeeze(-1)
            if self.use_quality:
                o['pred_quality_logits'] = self.dec_quality_head[layer_ids[j]](out_embeds[j]).squeeze(-1)
            if self.use_primary:
                o['pred_primary_logits'] = self.dec_primary_head[layer_ids[j]](out_embeds[j]).squeeze(-1)
            if self.use_unkobj:
                o['pred_unkobj_logits'] = self.dec_unkobj_head[layer_ids[j]](out_embeds[j]).squeeze(-1)
            if self.use_nc_obj:
                o['pred_nc_logits'] = self.dec_nc_head[layer_ids[j]](out_embeds[j]).squeeze(-1)
            if self.use_objectness:
                o['pred_obj'] = self.dec_obj_head[layer_ids[j]].mahalanobis(out_embeds[j])
                if self.training:
                    o['pred_embed'] = out_embeds[j]
                    o['obj_key'] = f'dec{layer_ids[j]}'
            outs.append(o)

        out = outs[-1]
        if getattr(self, 'return_embed', False) and not self.training:
            out['pred_embed'] = out_embeds[-1]   # used by post-hoc tools (E1 Gaussian fit, Stage A)
        if self.use_objectness and not self.training:
            out['pred_objectness'] = self.dec_obj_head[layer_ids[-1]].objectness(out['pred_obj'])
            out['obj_dim'] = self.hidden_dim

        if self.training and self.aux_loss:
            out['aux_outputs'] = outs[:-1]
            out['enc_aux_outputs'] = enc_aux
            out['enc_meta'] = {'class_agnostic': self.query_select_method == 'agnostic'}
            if dn_meta is not None:
                out['dn_aux_outputs'] = [{'pred_logits': a, 'pred_boxes': b}
                                         for a, b in zip(dn_out_logits, dn_out_bboxes)]
                out['dn_meta'] = dn_meta
            if o2m:
                out['o2m_outputs'] = [{'pred_logits': a, 'pred_boxes': b} for a, b in zip(o2m_logits, o2m_bboxes)]

        if self.training and (self.use_objectness or self.query_select_method == 'objectness'):
            heads = {}
            if self.use_objectness:
                heads.update({f'dec{i}': h for i, h in enumerate(self.dec_obj_head)})
            if self.query_select_method == 'objectness':
                heads['enc'] = self.enc_obj_head
            out['obj_heads'] = heads
        return out
