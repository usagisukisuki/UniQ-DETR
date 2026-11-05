"""Why do 'unknown' detections on known objects survive duplicate suppression? (docs/NEXT_STEPS.md)

For the top-r unknown detections of each COCO test image that hit a known GT g (IoU >= 0.5):
  S  the same query is also output as a known detection on g (self-duplicate; suppression skips self)
  A  g has no known-class detection in the top-100 (the known object is only reported as unknown)
  B  a known detection of g exists but IoU(unknown det, known det) < unk_dup_iou
  C  IoU >= unk_dup_iou but p(query, class of the known det) < unk_dup_cls_p
  D  otherwise (e.g. the known det's query has max p < unk_dup_tau)
Also: whether the known detection of g has the correct class, and the unknown query's p(class of g).

  uv run python tools/diag_dups.py -c configs/f1_quality_50ep.yml -r outputs/f1_quality_50ep/best.pth
"""
import argparse, json, os, sys
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, torchvision  # noqa: E401,E402
import owod.register  # noqa: F401,E402
from src.core import YAMLConfig  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('-c', '--config', required=True); ap.add_argument('-r', '--resume', required=True)
ap.add_argument('--top', type=int, default=5); ap.add_argument('--max-images', type=int, default=2000)
args = ap.parse_args()
cfg = YAMLConfig(args.config, val_dataloader={'total_batch_size': 16})
model = cfg.model; model.load_state_dict(torch.load(args.resume, map_location='cpu')['ema']['module'])
model = model.cuda().eval(); pp = cfg.postprocessor; K = cfg.yaml_cfg['owod_eval']['num_known']
t_iou = pp.unk_dup_iou or 0.7; rho = pp.unk_dup_cls_p or 0.0
gt = json.load(open(cfg.yaml_cfg['owod_eval']['gt_json']))
anns = defaultdict(list)
for a in gt['annotations']:
    anns[a['image_id']].append(a)
src = {i['id']: i['source'] for i in gt['images']}
iou = lambda a, b: torchvision.ops.box_iou(a, b) if len(a) and len(b) else torch.zeros(len(a), len(b))
cnt = defaultdict(float); pg = []
n = 0
with torch.no_grad():
    for x, tg in cfg.val_dataloader:
        keep = [j for j, t in enumerate(tg) if src[t['image_id'].item()] == 'coco']
        if not keep:
            continue
        out = model(x.cuda()); dets = pp(out, torch.stack([t['orig_size'] for t in tg]).cuda())
        P = out['pred_logits'].float().sigmoid().cpu()
        for j in keep:
            iid = tg[j]['image_id'].item(); d = {k: v.cpu() for k, v in dets[j].items()}
            G = [a for a in anns[iid] if a['category_id'] < K]
            if not G:
                n += 1; continue
            gb = torch.tensor([[a['bbox'][0], a['bbox'][1], a['bbox'][0] + a['bbox'][2], a['bbox'][1] + a['bbox'][3]] for a in G]).float()
            gc = torch.tensor([a['category_id'] for a in G])
            ui = (d['labels'] == K).nonzero().squeeze(1)[:args.top]
            ki = (d['labels'] < K).nonzero().squeeze(1)
            kb, kq, kl = d['boxes'][ki].float(), d['query_index'][ki], d['labels'][ki]
            for u in ui:
                ub = d['boxes'][u].float().unsqueeze(0); uq = int(d['query_index'][u])
                m = iou(ub, gb)[0]
                if m.max() < 0.5:
                    continue
                g = int(m.argmax()); cnt['n'] += 1
                pg.append(float(P[j, uq, gc[g]]))
                on_g = (iou(kb, gb[g:g + 1])[:, 0] >= 0.5) if len(kb) else torch.zeros(0, dtype=torch.bool)
                if not on_g.any():
                    cnt['A'] += 1; continue
                cnt['known_correct_cls'] += float((kl[on_g] == gc[g]).any())
                if (kq[on_g] == uq).any():
                    cnt['S'] += 1; continue
                best = iou(ub, kb[on_g])[0]
                bi = int(best.argmax()); oq = int(kq[on_g][bi]); oc = int(kl[on_g][bi])
                if best[bi] < t_iou:
                    cnt['B'] += 1
                elif float(P[j, uq, oc]) < rho:
                    cnt['C'] += 1
                else:
                    cnt['D'] += 1
            n += 1
        if n >= args.max_images:
            break
N = max(cnt['n'], 1)
print(f'images {n} | top-{args.top} unknown dets on a known GT: {cnt["n"]:.0f}  (dup IoU {t_iou}, cls p {rho})')
for k, desc in [('S', 'same query also output as known on that object (self-duplicate)'),
                ('A', 'object has no known-class detection in top-100'),
                ('B', f'known det exists, IoU(unknown, known) < {t_iou}'),
                ('C', f'IoU >= {t_iou} but p(query, known class) < {rho}'),
                ('D', 'other')]:
    print(f'  {k} {100 * cnt[k] / N:5.1f}%  {desc}')
print(f'  (known det on the object has the correct class in {100 * cnt["known_correct_cls"] / max(N - cnt["A"], 1):.1f}% of non-A cases)')
pg = torch.tensor(pg)
print(f'  unknown query p(class of the GT): median {pg.median():.3f}, >=0.3 {100 * (pg >= 0.3).float().mean():.1f}%')
