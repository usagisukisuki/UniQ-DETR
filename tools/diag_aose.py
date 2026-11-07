"""Where does A-OSE come from? (analysis/G4_PLAN.md 6.6, G4-8 vs G4-6)

A-OSE = number of known-class detections in the top-100 that overlap an unknown GT (IoU > 0.5), at any score.
For those detections (COCO-source test images): score bins, rank in the top-100, whether the same query also puts
an 'unknown' entry into the top-100, whether the unknown GT is hit by some unknown detection, and the unknown GT size.
Also per image: known / unknown detections and distinct queries in the top-100.

  uv run python tools/diag_aose.py -c configs/uniq_detr.yml -r outputs/g4_8/best.pth
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

BINS = (0.02, 0.05, 0.1, 0.3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-c', '--config', required=True)
    ap.add_argument('-r', '--resume', required=True)
    ap.add_argument('--nq', type=int, default=None)
    ap.add_argument('--w', type=float, default=None)
    args = ap.parse_args()
    over = {'val_dataloader': {'total_batch_size': 16}}
    if args.nq:
        over['OWRTDETRTransformerv2'] = {'num_queries': args.nq}
    if args.w is not None:
        over['OWPostProcessor'] = {'unk_weight': args.w}
    cfg = YAMLConfig(args.config, **over)
    model = cfg.model
    model.load_state_dict(torch.load(args.resume, map_location='cpu')['ema']['module'])
    model = model.cuda().eval()
    pp, K = cfg.postprocessor, cfg.yaml_cfg['owod_eval']['num_known']
    gt = json.load(open(cfg.yaml_cfg['owod_eval']['gt_json']))
    anns = defaultdict(list)
    for a in gt['annotations']:
        anns[a['image_id']].append(a)
    src = {i['id']: i['source'] for i in gt['images']}
    iou = lambda a, b: torchvision.ops.box_iou(a, b) if len(a) and len(b) else torch.zeros(len(a), len(b))
    box = lambda L: torch.tensor([[a['bbox'][0], a['bbox'][1], a['bbox'][0] + a['bbox'][2], a['bbox'][1] + a['bbox'][3]]
                                  for a in L], dtype=torch.float32).reshape(-1, 4)
    ose = []                     # score, rank, pair, unk GT hit by unknown det, small, medium, ose dets on same GT
    per_img = []                 # known dets, unknown dets, distinct queries
    n = 0
    with torch.no_grad():
        for x, tg in cfg.val_dataloader:
            keep = [j for j, t in enumerate(tg) if src[t['image_id'].item()] == 'coco']
            if not keep:
                continue
            out = model(x.cuda())
            sizes = torch.stack([t['orig_size'] for t in tg]).cuda()
            dets = pp(out, sizes)
            for j in keep:
                d = {k: v.cpu() for k, v in dets[j].items()}
                lab, sc, bx, q = d['labels'], d['scores'].float(), d['boxes'].float(), d['query_index']
                isk = lab < K
                per_img.append([int(isk.sum()), int((~isk).sum()), len(q.unique())])
                U = [a for a in anns[tg[j]['image_id'].item()] if a['category_id'] >= K]
                if not U:
                    continue
                gu = box(U)
                ov = iou(bx, gu)
                unk_hit = (ov[~isk] >= 0.5).any(0) if (~isk).any() else torch.zeros(len(U), dtype=torch.bool)
                unk_q = set(q[~isk].tolist())
                per_gt = defaultdict(int)
                for i in torch.where(isk & (ov.max(1).values > 0.5))[0].tolist():
                    g = int(ov[i].argmax())
                    per_gt[g] += 1
                    ar = U[g]['bbox'][2] * U[g]['bbox'][3]
                    ose.append([sc[i].item(), i, q[i].item() in unk_q, bool(unk_hit[g]), ar < 32 ** 2,
                                32 ** 2 <= ar < 96 ** 2, g + 1000 * len(per_img)])
            n += len(keep)
    O = torch.tensor(ose, dtype=torch.float64)
    P = torch.tensor(per_img, dtype=torch.float32)
    print(f'{os.path.basename(args.config)} {args.resume} nq {args.nq} w {args.w} | COCO images {n}')
    print(f'  per image: known dets {P[:, 0].mean():.1f}, unknown dets {P[:, 1].mean():.1f}, '
          f'distinct queries {P[:, 2].mean():.1f}')
    print(f'  OSE dets {len(O)} ({len(O) / n:.2f}/img) on {len(O[:, 6].unique())} unknown GT')
    edges = (0.,) + BINS + (1e9,)
    print('  score bins: ' + '  '.join(f'[{lo:g},{hi:g}) {100 * ((O[:, 0] >= lo) & (O[:, 0] < hi)).double().mean():.0f}%'
                                       for lo, hi in zip(edges[:-1], edges[1:])))
    print(f'  rank in top-100: median {O[:, 1].median():.0f}, >=50 {100 * (O[:, 1] >= 50).double().mean():.0f}%')
    print(f'  same query also in top-100 as unknown {100 * O[:, 2].mean():.0f}% | unknown GT also hit by an unknown det '
          f'{100 * O[:, 3].mean():.0f}% | GT small / medium / large {100 * O[:, 4].mean():.0f} / '
          f'{100 * O[:, 5].mean():.0f} / {100 * (1 - O[:, 4] - O[:, 5]).mean():.0f}%')


if __name__ == '__main__':
    main()
