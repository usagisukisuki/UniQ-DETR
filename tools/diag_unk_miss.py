"""Why are covered unknown objects missed? (analysis/G4_PLAN.md 6.6, plan C)

COCO-source test images. For each unknown GT: covered = some decoder query box has IoU >= 0.5 with it; hit = an
'unknown' detection in the top-100 has IoU >= 0.5. For covered GT, the 'best' covering query is the one with the highest
unknown score. Prints
  * coverage / hit rate by GT size (COCO small / medium / large) and by overlap with a known GT
  * for covered-and-hit vs covered-but-missed GT: the best covering query's unknown-score rank in the image and its
    signals (o, sigma(u), q, sigma(r), area, max known p), as medians
  * the same signals for the unknown detections in the top-100 that hit nothing (what takes the slots)

  uv run python tools/diag_unk_miss.py -c configs/g4_4_nq900.yml -r outputs/g4_4/best.pth \
      --pp unk_area_exp=0.25 unk_r_gamma=0.5 unk_q_exp=1 unk_weight=12
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

SIG = ['o', 'su', 'q', 'sr', 'area', 'maxp']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-c', '--config', required=True)
    ap.add_argument('-r', '--resume', required=True)
    ap.add_argument('--pp', nargs='*', default=[], help='OWPostProcessor overrides')
    ap.add_argument('--max-images', type=int, default=2000)
    args = ap.parse_args()
    pp_over = {k: float(v) for k, v in (a.split('=') for a in args.pp)}
    cfg = YAMLConfig(args.config, OWPostProcessor=pp_over, val_dataloader={'total_batch_size': 16})
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
    by_size = defaultdict(lambda: [0, 0, 0])       # n, covered, hit
    by_ov = defaultdict(lambda: [0, 0, 0])
    rows = {'hit': [], 'miss': [], 'fp': []}       # [rank, signals...]
    n = 0
    with torch.no_grad():
        for x, tg in cfg.val_dataloader:
            keep = [j for j, t in enumerate(tg) if src[t['image_id'].item()] == 'coco']
            if not keep:
                continue
            out = model(x.cuda())
            sizes = torch.stack([t['orig_size'] for t in tg]).cuda()
            dets = pp(out, sizes)
            sc = pp.scores(out)[..., -1].float()
            pb = out['pred_boxes'].float()
            area = pb[..., 2] * pb[..., 3]
            unk = sc * area.clamp(min=1e-6) ** pp.unk_area_exp
            maxp = out['pred_logits'][..., :K].float().sigmoid().max(-1).values
            sig = torch.stack([out['pred_objectness'].float(), out['pred_unk_logits'].float().sigmoid(),
                               out['pred_quality_logits'].float().sigmoid(), out['pred_primary_logits'].float().sigmoid(),
                               area, maxp], -1).cpu()                                   # [B, Q, 6]
            rank = unk.argsort(-1, descending=True).argsort(-1).cpu()                # [B, Q] 0 = top
            for j in keep:
                iid = tg[j]['image_id'].item()
                w, h = tg[j]['orig_size'].tolist()
                A = [a for a in anns[iid] if a['category_id'] >= K]
                gu, gk = box(A), box([a for a in anns[iid] if a['category_id'] < K])
                qb = torchvision.ops.box_convert(pb[j].cpu(), 'cxcywh', 'xyxy') * torch.tensor([w, h, w, h])
                d = dets[j]
                um = d['labels'] == K
                ub, uq = d['boxes'][um].float().cpu(), d['query_index'][um].cpu()
                hit = (iou(ub, gu) >= 0.5).any(0) if len(ub) else torch.zeros(len(gu), dtype=torch.bool)
                if len(ub):
                    fp = ~(iou(ub, gu) >= 0.5).any(1) if len(gu) else torch.ones(len(ub), dtype=torch.bool)
                    for q in uq[fp].tolist():
                        rows['fp'].append([rank[j, q].item()] + sig[j, q].tolist())
                cov = iou(gu, qb) >= 0.5                                                 # [G, Q]
                kov = iou(gu, gk).max(1).values if len(gk) else torch.zeros(len(gu))
                for g, a in enumerate(A):
                    ar = a['bbox'][2] * a['bbox'][3]
                    sz = 'small' if ar < 32 ** 2 else 'medium' if ar < 96 ** 2 else 'large'
                    ov = 'known IoU>=0.3' if kov[g] >= 0.3 else 'known IoU<0.3'
                    c, hh = bool(cov[g].any()), bool(hit[g])
                    for t, k in ((by_size, sz), (by_ov, ov)):
                        t[k][0] += 1; t[k][1] += c; t[k][2] += hh
                    if c:
                        qs = torch.where(cov[g])[0]
                        q = qs[rank[j, qs].argmin()].item()
                        rows['hit' if hh else 'miss'].append([rank[j, q].item()] + sig[j, q].tolist())
                n += 1
            if n >= args.max_images:
                break
    print(f'{os.path.basename(args.config)} {args.pp} | images {n}')
    for name, t in (('size', by_size), ('overlap', by_ov)):
        for k in sorted(t):
            m, c, hh = t[k]
            print(f'  {name:7s} {k:15s} GT {m:5d} | coverage {100 * c / m:5.1f}% | hit {100 * hh / m:5.1f}% | '
                  f'hit/covered {100 * hh / max(c, 1):5.1f}%')
    print('  medians (rank = unknown-score rank in the image, 0 = top):')
    print('    ' + f'{"":28s} {"n":>6s} {"rank":>6s} ' + ' '.join(f'{s:>7s}' for s in SIG))
    for k, lab in (('hit', 'covered & hit (best query)'), ('miss', 'covered & missed (best q.)'),
                   ('fp', 'top-100 unknown, no GT')):
        if not rows[k]:
            continue
        r = torch.tensor(rows[k])
        med = r.median(0).values
        print(f'    {lab:28s} {len(r):6d} {med[0]:6.0f} ' + ' '.join(f'{v:7.3f}' for v in med[1:].tolist()))
    if rows['miss']:
        r = torch.tensor(rows['miss'])[:, 0]
        print(f'  missed best-query rank: <50 {100 * (r < 50).float().mean():.0f}%, 50-99 '
              f'{100 * ((r >= 50) & (r < 100)).float().mean():.0f}%, 100-299 {100 * ((r >= 100) & (r < 300)).float().mean():.0f}%, '
              f'>=300 {100 * (r >= 300).float().mean():.0f}%')


if __name__ == '__main__':
    main()
