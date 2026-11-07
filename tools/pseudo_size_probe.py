"""Pseudo-unknown quality by box size (analysis/G4_PLAN.md 6.6, after G4-7): from a make_pseudo_unk.py --dump on the
V1 probe, precision (box hits an unknown GT at IoU 0.5) per size bin of rank 1-5 / 6-10 / 11-15, and the recall of rules
'top-5 + ranks 6..k only if area >= min_area'.

  uv run python tools/pseudo_size_probe.py --ann data/v1_probe/coco_train_probe3000.json --dump outputs/pseudo_sim/e2ep2.pt
"""
import argparse
import json
from collections import defaultdict

import torch
import torchvision

BINS = [('small <32²', 0, 32 ** 2), ('medium', 32 ** 2, 96 ** 2), ('large ≥96²', 96 ** 2, 1e12)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ann', required=True)
    ap.add_argument('--dump', required=True)
    ap.add_argument('--num-known', type=int, default=20)
    args = ap.parse_args()
    gt = json.load(open(args.ann))
    gu = defaultdict(list)
    for a in gt['annotations']:
        if a['category_id'] >= args.num_known:
            x, y, w, h = a['bbox']
            gu[a['image_id']].append([x, y, x + w, y + h])
    gu = {i: torch.tensor(v, dtype=torch.float32) for i, v in gu.items()}
    res = torch.load(args.dump)
    n_unk = sum(len(g) for g in gu.values())
    print(f'{args.dump}: images {len(res)}, unknown GT {n_unk}')

    # precision per rank group x size bin
    stat = defaultdict(lambda: [0, 0])
    for iid, (b, *_) in res.items():
        g = gu.get(iid, torch.zeros(0, 4))
        hit = (torchvision.ops.box_iou(b, g) >= 0.5).any(1) if len(b) and len(g) else torch.zeros(len(b), dtype=torch.bool)
        area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        for j in range(min(len(b), 15)):
            grp = 'rank 1-5' if j < 5 else 'rank 6-10' if j < 10 else 'rank 11-15'
            for name, lo, hi in BINS:
                if lo <= area[j] < hi:
                    stat[(grp, name)][0] += 1
                    stat[(grp, name)][1] += int(hit[j])
    print('precision by rank and box size (boxes, precision %, share of the group):')
    for grp in ('rank 1-5', 'rank 6-10', 'rank 11-15'):
        tot = sum(stat[(grp, n)][0] for n, *_ in BINS)
        print(f'  {grp:10s} ' + '  '.join(f'{n}: {stat[(grp, n)][0]:6d} {100 * stat[(grp, n)][1] / max(stat[(grp, n)][0], 1):5.1f}% '
                                          f'({100 * stat[(grp, n)][0] / max(tot, 1):4.1f}%)' for n, *_ in BINS))

    print('rules (boxes / image, precision %, recall %):')

    def ev(name, sel_fn):
        kept = hit = 0
        rec = 0
        for iid, (b, *_) in res.items():
            s = sel_fn(b)
            g = gu.get(iid, torch.zeros(0, 4))
            kept += len(s)
            if len(s) and len(g):
                on = torchvision.ops.box_iou(s, g) >= 0.5
                hit += int(on.any(1).sum()); rec += int(on.any(0).sum())
        print(f'  {name:36s} {kept / len(res):5.2f}  {100 * hit / max(kept, 1):5.1f}  {100 * rec / n_unk:5.1f}')

    area = lambda b: (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    for k in (5, 10, 15):
        ev(f'top-{k}', lambda b, k=k: b[:k])
    for k in (5, 10):
        for m in (32, 64, 96):
            ev(f'top-{k}, area >= {m}²', lambda b, k=k, m=m: b[:k][area(b[:k]) >= m * m])
    for k in (10, 15, 20):
        for m in (32, 64, 96):
            ev(f'top-5 + rank 6-{k} with area >= {m}²',
               lambda b, k=k, m=m: torch.cat([b[:5], b[5:k][area(b[5:k]) >= m * m]]))


if __name__ == '__main__':
    main()
