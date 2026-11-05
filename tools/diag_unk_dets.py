"""What do the 'unknown' detections in the top-100 land on? (docs/NEXT_STEPS.md, duplicate hypothesis)

For each unknown-labelled detection (config's postprocessor) on COCO test images, by the box it hits best:
  unknown  IoU >= 0.5 with an unknown GT
  dup      IoU >= 0.5 with a known GT (a duplicate of a known object)
  frag     0.1 <= IoU < 0.5 with a known GT (part / loose box around a known object), and not 'unknown'
  ufrag    0.1 <= IoU < 0.5 with an unknown GT (badly localised unknown), and none of the above
  bg       none of the above
reported for the top-r unknown detections of each image (r = 5, 10, 20, all).

  uv run python tools/diag_unk_dets.py -c configs/e2_obj.yml -r outputs/e2_obj/best.pth
"""
import argparse, json, os, sys
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, torchvision  # noqa: E401,E402
import owod.register  # noqa: F401,E402
from src.core import YAMLConfig  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('-c', '--config', required=True); ap.add_argument('-r', '--resume', required=True)
ap.add_argument('--max-images', type=int, default=2000)
args = ap.parse_args()
cfg = YAMLConfig(args.config, val_dataloader={'total_batch_size': 16})
model = cfg.model; model.load_state_dict(torch.load(args.resume, map_location='cpu')['ema']['module'])
model = model.cuda().eval(); pp = cfg.postprocessor; K = cfg.yaml_cfg['owod_eval']['num_known']
gt = json.load(open(cfg.yaml_cfg['owod_eval']['gt_json']))
anns = defaultdict(list)
for a in gt['annotations']:
    anns[a['image_id']].append(a)
src = {i['id']: i['source'] for i in gt['images']}
box = lambda L: torch.tensor([[a['bbox'][0], a['bbox'][1], a['bbox'][0] + a['bbox'][2], a['bbox'][1] + a['bbox'][3]]
                              for a in L], dtype=torch.float32).reshape(-1, 4)
iou = lambda a, b: torchvision.ops.box_iou(a, b) if len(a) and len(b) else torch.zeros(len(a), len(b))
R = [5, 10, 20, 100]
cnt = {r: defaultdict(float) for r in R}
n = 0
with torch.no_grad():
    for x, tg in cfg.val_dataloader:
        keep = [j for j, t in enumerate(tg) if src[t['image_id'].item()] == 'coco']
        if not keep:
            continue
        out = model(x.cuda()); dets = pp(out, torch.stack([t['orig_size'] for t in tg]).cuda())
        for j in keep:
            iid = tg[j]['image_id'].item(); d = dets[j]
            ub = d['boxes'][d['labels'] == K].float().cpu()       # already sorted by score
            gk = box([a for a in anns[iid] if a['category_id'] < K]); gu = box([a for a in anns[iid] if a['category_id'] >= K])
            mk = iou(ub, gk).max(1).values if len(gk) else torch.zeros(len(ub))
            mu = iou(ub, gu).max(1).values if len(gu) else torch.zeros(len(ub))
            cat = ['unknown' if u >= 0.5 else 'dup' if k >= 0.5 else 'frag' if k >= 0.1 else 'ufrag' if u >= 0.1 else 'bg' for u, k in zip(mu, mk)]
            for r in R:
                for c in cat[:r]:
                    cnt[r][c] += 1
                cnt[r]['n'] += min(r, len(cat))
            n += 1
        if n >= args.max_images:
            break
print(f'images {n}')
for r in R:
    c = cnt[r]; t = max(c['n'], 1)
    print(f'top-{r:3d} unknown dets/image ({c["n"]:.0f}): unknown {100*c["unknown"]/t:5.1f}%  dup-of-known {100*c["dup"]/t:5.1f}%  '
          f'frag-of-known {100*c["frag"]/t:5.1f}%  loose-on-unknown {100*c["ufrag"]/t:5.1f}%  bg {100*c["bg"]/t:5.1f}%')
