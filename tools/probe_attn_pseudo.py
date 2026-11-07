"""Can the detector's own attention rank pseudo-unknown candidates better than flip consistency? (analysis/G4_PLAN.md 6.6)

Candidates are those of tools/make_pseudo_unk.py (teacher decoder boxes with IoU < 0.5 to every known GT, flip
consistency 'cons'); every candidate also gets attention signals from the same forward pass:

  (1) encoder self-attention (AIFI on the stride-32 map, heads averaged, A [N, N] rows sum to 1); tokens of a box =
      tokens whose centre is inside it (at least the token under the box centre)
      e_recv   mean attention the box's tokens receive (sum over rows), 1 = average token
      e_self   mean over the box's tokens of the attention mass that stays inside the box, minus the box's token share
               (an object attends to itself more than its size predicts; normalised-cut style)
  (2) decoder cross-attention of the last layer for the candidate's own query (deformable: 8 heads x 12 points)
      d_in     attention weight on sampling points inside the query's predicted box (heads averaged)
      d_core   the same for the central half of the box
      d_ent    1 - mean normalised entropy of the point weights per head (1 = all weight on one point)
      d_fine   attention weight on the stride-8 level

Prints, on a probe with unknown GT (hit = IoU >= 0.5 to an unknown GT): AUROC of each signal over all candidates and
over cons-ranked top-50 after NMS, medians for hit / non-hit, and top-k precision / recall when ranking by the signal,
or by cons x the signal's per-image percentile.

  uv run python tools/probe_attn_pseudo.py -c configs/e2_obj.yml -r outputs/e2_pilot_bn/last.pth \
      --ann data/v1_probe/coco_train_probe3000.json
"""
import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
import torchvision  # noqa: E402

import owod.register  # noqa: F401,E402
from src.core import YAMLConfig  # noqa: E402

SIG = ['cons', 'e_recv', 'e_self', 'd_in', 'd_core', 'd_ent', 'd_fine']


def iou(a, b):
    return torchvision.ops.box_iou(a, b) if len(a) and len(b) else torch.zeros(len(a), len(b))


def boxes_of(anns):
    return torch.tensor([[a['bbox'][0], a['bbox'][1], a['bbox'][0] + a['bbox'][2], a['bbox'][1] + a['bbox'][3]]
                         for a in anns], dtype=torch.float32).reshape(-1, 4)


def auroc(s, y):
    if y.all() or (~y).all():
        return float('nan')
    r = s.argsort().argsort().float() + 1
    n1, n0 = y.sum().float(), (~y).sum().float()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


class Taps:
    """Captures the encoder self-attention weights and the last decoder layer's deformable sampling."""

    def __init__(self, model):
        self.enc = self.loc = self.w = None
        attn = model.encoder.encoder[-1].layers[-1].self_attn

        def pre(mod, args, kwargs):
            kwargs['need_weights'] = True
            kwargs['average_attn_weights'] = True
            return args, kwargs

        def post(mod, args, kwargs, out):
            self.enc = out[1].float()                                    # [B, N, N]
        attn.register_forward_pre_hook(pre, with_kwargs=True)
        attn.register_forward_hook(post, with_kwargs=True)
        ca = model.decoder.decoder.layers[-1].cross_attn
        core = ca.ms_deformable_attn_core
        self.npl = ca.num_points_list

        def wrapped(value, shapes, loc, w, npl):
            self.loc, self.w = loc.float(), w.float()                    # [B, Q, H, P, 2], [B, Q, H, P]
            return core(value, shapes, loc, w, npl)
        ca.ms_deformable_attn_core = wrapped


def enc_signals(A, bn, hw):
    """A [N, N] row-stochastic, bn [n, 4] normalised xyxy -> e_recv, e_self [n]."""
    h, w = hw
    ys, xs = torch.meshgrid((torch.arange(h, device=A.device) + 0.5) / h, (torch.arange(w, device=A.device) + 0.5) / w,
                            indexing='ij')
    xs, ys = xs.flatten(), ys.flatten()
    m = ((xs[None] >= bn[:, None, 0]) & (xs[None] <= bn[:, None, 2]) &
         (ys[None] >= bn[:, None, 1]) & (ys[None] <= bn[:, None, 3])).float()          # [n, N]
    cx = ((bn[:, 0] + bn[:, 2]) / 2 * w).long().clamp(0, w - 1)
    cy = ((bn[:, 1] + bn[:, 3]) / 2 * h).long().clamp(0, h - 1)
    empty = m.sum(1) == 0
    m[empty, (cy * w + cx)[empty]] = 1
    cnt = m.sum(1)
    recv = A.sum(0)                                                                     # [N]
    e_recv = (m @ recv) / cnt
    inside = (m @ A * m).sum(1) / cnt                                                   # mean row mass inside the box
    e_self = inside - cnt / (h * w)
    return e_recv, e_self


def dec_signals(loc, w, bn, npl):
    """loc [n, H, P, 2] (x, y normalised), w [n, H, P], bn [n, 4] normalised xyxy."""
    x, y = loc[..., 0], loc[..., 1]
    b = bn[:, None, None]

    def frac(bx):
        ins = (x >= bx[..., 0]) & (x <= bx[..., 2]) & (y >= bx[..., 1]) & (y <= bx[..., 3])
        return (w * ins).sum(-1).mean(-1)
    cxy = torch.stack([(bn[:, 0] + bn[:, 2]) / 2, (bn[:, 1] + bn[:, 3]) / 2], -1)
    half = torch.stack([bn[:, 2] - bn[:, 0], bn[:, 3] - bn[:, 1]], -1) / 4
    core = torch.cat([cxy - half, cxy + half], -1)[:, None, None]
    P = w.shape[-1]
    ent = -(w.clamp(min=1e-9) * w.clamp(min=1e-9).log()).sum(-1) / torch.log(torch.tensor(float(P)))
    return frac(b), frac(core), 1 - ent.mean(-1), w[..., :npl[0]].sum(-1).mean(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-c', '--config', required=True)
    ap.add_argument('-r', '--resume', required=True)
    ap.add_argument('--ann', required=True)
    ap.add_argument('--bs', type=int, default=16)
    ap.add_argument('--num-known', type=int, default=None)
    args = ap.parse_args()
    cfg = YAMLConfig(args.config, val_dataloader={'total_batch_size': args.bs, 'shuffle': False,
                                                  'dataset': {'ann_file': args.ann, 'split': 'test'}})
    model = cfg.model
    model.load_state_dict(torch.load(args.resume, map_location='cpu')['ema']['module'])
    model = model.cuda().eval()
    taps = Taps(model)
    K = args.num_known if args.num_known is not None else cfg.yaml_cfg['num_classes']
    gt = json.load(open(args.ann))
    anns = defaultdict(list)
    for a in gt['annotations']:
        anns[a['image_id']].append(a)
    n_unk = sum(1 for a in gt['annotations'] if a['category_id'] >= K)
    res = {}
    with torch.no_grad():
        for x, tg in cfg.val_dataloader:
            x = x.cuda()
            outf = model(x.flip(-1))
            out = model(x)                                                   # taps hold this (un-flipped) pass
            hw = (x.shape[2] // 32, x.shape[3] // 32)
            for j, t in enumerate(tg):
                iid = t['image_id'].item()
                W, H = t['orig_size'].tolist()
                scale = torch.tensor([W, H, W, H], dtype=torch.float32)
                bn = torchvision.ops.box_convert(out['pred_boxes'][j].float(), 'cxcywh', 'xyxy')   # normalised, GPU
                db = bn.cpu() * scale
                fb = outf['pred_boxes'][j].float().clone()
                fb[:, 0] = 1 - fb[:, 0]
                fb = torchvision.ops.box_convert(fb.cpu(), 'cxcywh', 'xyxy') * scale
                gk = boxes_of([a for a in anns[iid] if a['category_id'] < K])
                free = (iou(db, gk).max(1).values < 0.5) if len(gk) else torch.ones(len(db), dtype=torch.bool)
                if not free.any():
                    continue
                cons = iou(db[free], fb).max(1).values
                fg = free.cuda()
                e_recv, e_self = enc_signals(taps.enc[j], bn[fg], hw)
                d = dec_signals(taps.loc[j][fg], taps.w[j][fg], bn[fg], taps.npl)
                s = torch.stack([cons, e_recv.cpu(), e_self.cpu()] + [v.cpu() for v in d], -1)   # [n, len(SIG)]
                gu = boxes_of([a for a in anns[iid] if a['category_id'] >= K])
                ov = iou(db[free], gu) >= 0.5
                res[iid] = (db[free], s, ov)
    print(f'{args.resume} on {args.ann}: images {len(res)}, unknown GT {n_unk}')

    def pct(v):
        return v.argsort().argsort().float() / max(len(v) - 1, 1)

    allS = torch.cat([s for _, s, _ in res.values()])
    allY = torch.cat([o.any(1) for *_, o in res.values()])
    top = [(b, s, o) for b, s, o in res.values()]
    selS, selY = [], []
    for b, s, o in top:
        keep = torchvision.ops.nms(b, s[:, 0] + 1e-9, 0.5)[:50]
        selS.append(s[keep]); selY.append(o[keep].any(1))
    selS, selY = torch.cat(selS), torch.cat(selY)
    print(f'candidates {len(allY)} (hit {100 * allY.float().mean():.1f}%), cons top-50 after NMS {len(selY)} '
          f'(hit {100 * selY.float().mean():.1f}%)')
    print(f'  {"signal":8s} {"AUROC all":>9s} {"AUROC top50":>11s} {"median hit":>10s} {"median miss":>11s}')
    for i, n in enumerate(SIG):
        print(f'  {n:8s} {auroc(allS[:, i], allY):9.3f} {auroc(selS[:, i], selY):11.3f} '
              f'{selS[selY, i].median():10.3f} {selS[~selY, i].median():11.3f}')

    def rank_eval(name, score_fn):
        out = []
        for k in (5, 10):
            kept = hit = rec = 0
            for b, s, o in top:
                sc = score_fn(s)
                keep = torchvision.ops.nms(b, sc + 1e-9, 0.5)[:k]
                kept += len(keep)
                hit += int(o[keep].any(1).sum())
                rec += int(o[keep].any(0).sum())
            out.append(f'k {k}: {kept / len(top):4.2f}/img {100 * hit / max(kept, 1):5.1f} / {100 * rec / n_unk:5.1f}')
        print(f'  {name:22s} ' + '   '.join(out))
    print('ranking (boxes / image, precision % / recall %):')
    rank_eval('cons (current)', lambda s: s[:, 0])
    for i, n in enumerate(SIG[1:], 1):
        rank_eval(n, lambda s, i=i: s[:, i])
        rank_eval(f'cons x pct({n})', lambda s, i=i: s[:, 0] * pct(s[:, i]))


if __name__ == '__main__':
    main()
