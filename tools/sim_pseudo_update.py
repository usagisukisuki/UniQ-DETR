"""Simulate pseudo-unknown label update rules offline (analysis/G4_PLAN.md 6.6).

Input: per-checkpoint candidate dumps from tools/make_pseudo_unk.py --dump (top-50 per image after the known-GT filter
and NMS 0.5, with flip consistency 'cons' and max known p 'maxp'). The first dump is the anchor (the labels used now,
E2 epoch 2); the others are the teachers in training order (e.g. G4-4 epochs 9..49), standing in for 'the model being
trained at each epoch'. Every rule is scored on the V1 probe's unknown GT (precision / recall at IoU 0.5, as
make_pseudo_unk.py), so a rule beats the current labels if it has higher precision at the same boxes / image.

Rules
  fixed k           anchor top-k (current labels)
  replace k         the last teacher's top-k (round-2 labels)
  ema m k           track boxes over [anchor, teachers]: a new candidate matched (IoU >= 0.7) to a track updates
                    s <- m*s + (1-m)*cons and the box likewise; unmatched tracks decay s <- m*s; new ones start at (1-m)*cons
  add n tau a       anchor top-5 + up to a boxes per image seen (IoU >= 0.7) with cons >= tau in >= n teachers and
                    IoU < 0.5 to every anchor box, ranked by mean cons
  drop n tau        anchor top-k minus boxes where >= n teachers have a matching (IoU >= 0.7) box with maxp >= tau
                    (the model has come to call it a known class)
  unstable n tau    anchor top-k minus boxes that < n teachers reproduce with cons >= tau

  uv run python tools/sim_pseudo_update.py --ann data/v1_probe/coco_train_probe3000.json \
      --dumps outputs/pseudo_sim/e2ep2.pt outputs/pseudo_sim/g4_4_ep00{09,19,29,39,49}.pt
"""
import argparse
import json
from collections import defaultdict

import torch
import torchvision

K = 20


def iou(a, b):
    return torchvision.ops.box_iou(a, b) if len(a) and len(b) else torch.zeros(len(a), len(b))


def evaluate(name, sel, gu):
    kept = hit = rec = 0.
    for iid, b in sel.items():
        kept += len(b)
        g = gu.get(iid, torch.zeros(0, 4))
        if len(g) and len(b):
            on = iou(b, g) >= 0.5
            hit += float(on.any(1).sum())
            rec += float(on.any(0).sum())
    n_unk = sum(len(g) for g in gu.values())
    print(f'  {name:34s} boxes/image {kept / len(sel):5.2f}  precision {100 * hit / max(kept, 1):5.1f}  '
          f'recall {100 * rec / max(n_unk, 1):5.1f}', flush=True)


def matched_any(b, cand, thr=0.7):
    """[len(b)] bool: b has a candidate with IoU >= thr; and the index of the best one."""
    if not len(b) or not len(cand):
        return torch.zeros(len(b), dtype=torch.bool), torch.zeros(len(b), dtype=torch.long)
    v, i = iou(b, cand).max(1)
    return v >= thr, i


def ema(anchor, teachers, iid, m, k):
    b, s = anchor[iid][0].clone(), anchor[iid][2].clone() * (1 - m)
    for t in teachers:
        cb, cc = t[iid][0], t[iid][2]
        s = s * m
        if len(cb) and len(b):
            ov = iou(b, cb)
            used = torch.zeros(len(cb), dtype=torch.bool)
            for r in torch.argsort(s, descending=True).tolist():
                ov_r = ov[r].clone()
                ov_r[used] = -1
                v, j = ov_r.max(0)
                if v >= 0.7:
                    used[j] = True
                    s[r] = s[r] + (1 - m) * cc[j]
                    b[r] = m * b[r] + (1 - m) * cb[j]
            new = ~used
        else:
            new = torch.ones(len(cb), dtype=torch.bool)
        b, s = torch.cat([b, cb[new]]), torch.cat([s, (1 - m) * cc[new]])
        keep = torchvision.ops.nms(b, s + 1e-9, 0.5)[:50]
        b, s = b[keep], s[keep]
    return b[:k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ann', required=True)
    ap.add_argument('--dumps', nargs='+', required=True)
    args = ap.parse_args()
    gt = json.load(open(args.ann))
    gu = defaultdict(list)
    for a in gt['annotations']:
        if a['category_id'] >= K:
            x, y, w, h = a['bbox']
            gu[a['image_id']].append([x, y, x + w, y + h])
    gu = {i: torch.tensor(v, dtype=torch.float32) for i, v in gu.items()}
    anchor, teachers = torch.load(args.dumps[0]), [torch.load(d) for d in args.dumps[1:]]
    ids = list(anchor)
    print(f'anchor {args.dumps[0]}, teachers {args.dumps[1:]}, images {len(ids)}')

    print('== fixed (current labels) / replace (last teacher)')
    for k in (3, 5, 7, 10, 15):
        evaluate(f'fixed k={k}', {i: anchor[i][0][:k] for i in ids}, gu)
    for k in (5, 10):
        for j, t in enumerate(teachers):
            evaluate(f'replace teacher{j} k={k}', {i: t[i][0][:k] for i in ids}, gu)

    print('== ema')
    for m in (0.5, 0.7, 0.9):
        for k in (5, 10):
            evaluate(f'ema m={m} k={k}', {i: ema(anchor, teachers, i, m, k) for i in ids}, gu)

    print('== add (anchor top-5 + boxes seen in >= n teachers)')
    for n in (2, 3, 5):
        for tau in (0.7, 0.85):
            for a in (2, 5):
                sel = {}
                for i in ids:
                    base = anchor[i][0][:5]
                    cand, cs = teachers[-1][i][0], teachers[-1][i][2]
                    cnt = torch.zeros(len(cand))
                    tot = torch.zeros(len(cand))
                    for t in teachers:
                        ok, j = matched_any(cand, t[i][0])
                        c = t[i][2][j] if len(t[i][2]) else torch.zeros(len(cand))
                        good = ok & (c >= tau)
                        cnt += good.float()
                        tot += torch.where(good, c, torch.zeros_like(c))
                    free = (iou(cand, base).max(1).values < 0.5) if len(base) else torch.ones(len(cand), dtype=torch.bool)
                    ok = (cnt >= n) & free
                    idx = torch.where(ok)[0]
                    idx = idx[torch.argsort(tot[idx] / cnt[idx], descending=True)][:a]
                    extra = cand[idx]
                    if len(extra):
                        extra = extra[torchvision.ops.nms(extra, (tot[idx] / cnt[idx]) + 1e-9, 0.5)]
                    sel[i] = torch.cat([base, extra])
                evaluate(f'add n={n} tau={tau} a={a}', sel, gu)

    print('== drop (anchor boxes the teachers call known) / unstable (anchor boxes the teachers do not reproduce)')
    for k in (5, 7, 10):
        for n in (1, 2, 3):
            for tau in (0.3, 0.5):
                sel = {}
                for i in ids:
                    b = anchor[i][0][:k]
                    cnt = torch.zeros(len(b))
                    for t in teachers:
                        ok, j = matched_any(b, t[i][0])
                        mp = t[i][3][j] if len(t[i][3]) else torch.zeros(len(b))
                        cnt += (ok & (mp >= tau)).float()
                    sel[i] = b[cnt < n]
                evaluate(f'drop k={k} n={n} tau={tau}', sel, gu)
        for n in (2, 3, 4):
            for tau in (0.5, 0.7):
                sel = {}
                for i in ids:
                    b = anchor[i][0][:k]
                    cnt = torch.zeros(len(b))
                    for t in teachers:
                        ok, j = matched_any(b, t[i][0])
                        c = t[i][2][j] if len(t[i][2]) else torch.zeros(len(b))
                        cnt += (ok & (c >= tau)).float()
                    sel[i] = b[cnt >= n]
                evaluate(f'unstable k={k} n={n} tau={tau}', sel, gu)


if __name__ == '__main__':
    main()
