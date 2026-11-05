"""How does each model reach its U-Recall? Unknown-output statistics at a given inference setting (COCO test images).

Per image (COCO-source test images, which carry unknown GT):
  n_unk      number of 'unknown' detections in the top-100
  precision  fraction of them with IoU >= 0.5 to an unknown GT
  recall     unique unknown GT hit by >= 1 unknown detection (approximate U-Recall, no 1-to-1 assignment)
  coverage   unknown GT covered (IoU >= 0.5) by any of the decoder's final query boxes (upper bound)
  rank       for each hit unknown GT, the rank of its best-scoring hitting detection among the image's unknown dets

  uv run python tools/diag_unk_output.py -c configs/baselines/owdetr_rt.yml -r outputs/b_owdetr_rt/best.pth --w 32
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-c', '--config', required=True)
    ap.add_argument('-r', '--resume', required=True)
    ap.add_argument('--w', type=float, required=True)
    ap.add_argument('--max-images', type=int, default=2000)
    args = ap.parse_args()
    cfg = YAMLConfig(args.config, OWPostProcessor={'unk_weight': args.w}, val_dataloader={'total_batch_size': 16})
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
    s = defaultdict(float)
    ranks = []
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
                iid = tg[j]['image_id'].item()
                w, h = tg[j]['orig_size'].tolist()
                gu = box([a for a in anns[iid] if a['category_id'] >= K])
                d = dets[j]
                um = d['labels'] == K
                ub = d['boxes'][um].float().cpu()
                s['n_unk'] += len(ub); s['img'] += 1; s['n_gt'] += len(gu)
                qb = torchvision.ops.box_convert(out['pred_boxes'][j].float().cpu(), 'cxcywh', 'xyxy') * torch.tensor([w, h, w, h])
                if len(gu):
                    s['cover'] += float((iou(gu, qb).max(1).values >= 0.5).sum())
                if len(ub) and len(gu):
                    m = iou(ub, gu) >= 0.5                    # [n_unk, n_gt], dets sorted by score
                    s['tp_det'] += float(m.any(1).sum())
                    hit = m.any(0)
                    s['hit'] += float(hit.sum())
                    first = torch.where(m, torch.arange(len(ub)).unsqueeze(1), torch.full_like(m, 10 ** 6, dtype=torch.long)).min(0).values
                    ranks += first[hit].tolist()
                n += 1
            if n >= args.max_images:
                break
    r = torch.tensor(ranks, dtype=torch.float32)
    print(f'{os.path.basename(args.config)} w={args.w:g} | images {n} | unknown dets/img {s["n_unk"] / s["img"]:.1f} | '
          f'precision {100 * s["tp_det"] / max(s["n_unk"], 1):.1f}% | recall {100 * s["hit"] / s["n_gt"]:.1f}% | '
          f'decoder coverage {100 * s["cover"] / s["n_gt"]:.1f}% | rank of hit: median {r.median():.0f}, '
          f'<10 {100 * (r < 10).float().mean():.0f}%, >=30 {100 * (r >= 30).float().mean():.0f}%')


if __name__ == '__main__':
    main()
