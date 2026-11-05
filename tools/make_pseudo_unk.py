"""G4-1 (analysis/G4_PLAN.md): offline pseudo-unknown boxes from a fixed early teacher.

Per image (val transform, no augmentation): teacher decoder boxes on the image and on its horizontal flip;
candidates = boxes with IoU < 0.5 to every known GT; score = flip consistency (max IoU with the un-flipped
flip prediction); class-agnostic NMS (IoU 0.5) on that score; keep the top k. Same rule as 'cons' in
tools/v1_pseudo_unk_quality.py. The NMS is only for building labels, never part of the detector's inference.

Two modes:
  * --ann has unknown GT (category >= num_known, e.g. the V1 probe): print precision / recall per k, write nothing else.
  * otherwise (the training file): write <out-prefix>_k{k}.json = the annotation file plus the pseudo boxes as
    annotations with category_id = num_known, source = 'pseudo', score = flip consistency (OWODDetection accepts these).

  uv run python tools/make_pseudo_unk.py -c configs/e2_obj.yml -r outputs/e2_pilot_bn/last.pth \
      --ann data/mowodb/t1_train.json --out-prefix data/pseudo_unk/e2ep2_cons
  # round 2 with an open-world teacher ranked by its own unknown score (analysis/G4_PLAN.md 6.6)
  uv run python tools/make_pseudo_unk.py -c configs/g4_4.yml -r outputs/g4_4/best.pth --num-known 20 --rank unk \
      --area-exp 0.25 --pp unk_r_gamma=0.5 --ann data/v1_probe/coco_train_probe3000.json --ks 5 10
"""
import argparse
import copy
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
import torchvision  # noqa: E402

import owod.register  # noqa: F401,E402
from src.core import YAMLConfig  # noqa: E402


def iou(a, b):
    return torchvision.ops.box_iou(a, b) if len(a) and len(b) else torch.zeros(len(a), len(b))


def boxes_of(anns):
    return torch.tensor([[a['bbox'][0], a['bbox'][1], a['bbox'][0] + a['bbox'][2], a['bbox'][1] + a['bbox'][3]]
                         for a in anns], dtype=torch.float32).reshape(-1, 4)


def predict(cfg, ckpt, gt, kmax, K=None, rank='cons', area_exp=0.0):
    """image_id -> (boxes xyxy in original pixels [n, 4], ranking score [n], flip consistency [n], max known p [n]),
    sorted by the ranking score, n <= kmax.

    rank: 'cons' = flip consistency; 'unk' = the teacher's own unknown score (OWPostProcessor.scores x area^area_exp,
    for an open-world teacher such as G4-4); 'cons_unk' = their product."""
    state = torch.load(ckpt, map_location='cpu')
    model = cfg.model
    model.load_state_dict(state['ema']['module'])
    model = model.cuda().eval()
    K = K if K is not None else cfg.yaml_cfg['num_classes']   # E2 teachers: 20 known classes, no unknown column
    pp = cfg.postprocessor
    anns = defaultdict(list)
    for a in gt['annotations']:
        anns[a['image_id']].append(a)
    res, n_img = {}, 0
    with torch.no_grad():
        for x, tg in cfg.val_dataloader:
            x = x.cuda()
            out = model(x)
            outf = model(x.flip(-1))
            for j, t in enumerate(tg):
                iid = t['image_id'].item()
                w, h = t['orig_size'].tolist()
                scale = torch.tensor([w, h, w, h], dtype=torch.float32)
                xyxy = lambda b: torchvision.ops.box_convert(b.float().cpu(), 'cxcywh', 'xyxy') * scale
                db = xyxy(out['pred_boxes'][j])
                fb = outf['pred_boxes'][j].clone()
                fb[:, 0] = 1 - fb[:, 0]                      # un-flip cx
                fb = xyxy(fb)
                gk = boxes_of([a for a in anns[iid] if a['category_id'] < K])
                free = (iou(db, gk).max(1).values < 0.5) if len(gk) else torch.ones(len(db), dtype=torch.bool)
                b = db[free]
                cons = iou(b, fb).max(1).values if len(b) else torch.zeros(0)
                fc = cons                                     # ranking score
                if rank != 'cons':
                    u = pp.scores({k: v[j:j + 1] for k, v in out.items() if torch.is_tensor(v)})[0, :, -1].float().cpu()
                    pb = out['pred_boxes'][j].float().cpu()
                    u = (u * (pb[:, 2] * pb[:, 3]).clamp(min=1e-6) ** area_exp)[free]
                    fc = u if rank == 'unk' else cons * u
                keep = torchvision.ops.nms(b, fc + 1e-9, 0.5)[:kmax] if len(b) else torch.zeros(0, dtype=torch.long)
                maxp = out['pred_logits'][j, :, :K].float().sigmoid().max(-1).values.cpu()[free]
                res[iid] = (b[keep], fc[keep], cons[keep], maxp[keep])
            n_img += len(tg)
            if n_img % 1600 < len(tg):
                print(f'  {n_img} images', flush=True)
    return res, K


def score(res, gt, K, ks):
    anns = defaultdict(list)
    for a in gt['annotations']:
        anns[a['image_id']].append(a)
    n_unk = sum(1 for a in gt['annotations'] if a['category_id'] >= K)
    for k in ks:
        kept = hit = rec = 0.
        for iid, (b, *_) in res.items():
            gu = boxes_of([a for a in anns[iid] if a['category_id'] >= K])
            sel = b[:k]
            kept += len(sel)
            if len(gu) and len(sel):
                on = iou(sel, gu) >= 0.5
                hit += float(on.any(1).sum())
                rec += float(on.any(0).sum())
        print(f'  k={k:2d}  boxes/image {kept / len(res):.2f}  precision {100 * hit / max(kept, 1):5.1f}  '
              f'recall {100 * rec / max(n_unk, 1):5.1f}  (images {len(res)}, unknown GT {n_unk})', flush=True)


def write(res, gt, K, ks, prefix, ckpt):
    os.makedirs(os.path.dirname(prefix), exist_ok=True)
    for k in ks:
        d = copy.deepcopy(gt)
        d.setdefault('info', {})['pseudo_unk'] = {'teacher': ckpt, 'rule': 'flip-consistency, NMS 0.5', 'k': k}
        nid = max(a['id'] for a in d['annotations']) + 1
        n = 0
        for iid, (b, c, *_) in res.items():
            for (x1, y1, x2, y2), s in zip(b[:k].tolist(), c[:k].tolist()):
                if x2 - x1 < 1 or y2 - y1 < 1:
                    continue
                d['annotations'].append({'id': nid, 'image_id': iid, 'category_id': K,
                                         'bbox': [x1, y1, x2 - x1, y2 - y1], 'area': (x2 - x1) * (y2 - y1),
                                         'iscrowd': 0, 'source': 'pseudo', 'score': s})
                nid += 1
                n += 1
        path = f'{prefix}_k{k}.json'
        json.dump(d, open(path, 'w'))
        print(f'  wrote {path}: {n} pseudo boxes on {len(res)} images ({n / max(len(res), 1):.2f}/image)', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-c', '--config', required=True)
    ap.add_argument('-r', '--resume', required=True)
    ap.add_argument('--ann', required=True)
    ap.add_argument('--ks', type=int, nargs='+', default=[5, 10])
    ap.add_argument('--out-prefix', default=None)
    ap.add_argument('--bs', type=int, default=16)
    ap.add_argument('--num-known', type=int, default=None, help='known classes (default: num_classes; 20 for G4 teachers)')
    ap.add_argument('--rank', default='cons', choices=['cons', 'unk', 'cons_unk'])
    ap.add_argument('--area-exp', type=float, default=0.0, help="unknown score x area^a for --rank unk / cons_unk")
    ap.add_argument('--dump', default=None, help='save {image_id: (boxes, score, flip cons, max known p)} (top max(ks)) '
                    'for tools/sim_pseudo_update.py instead of scoring / writing')
    ap.add_argument('--pp', nargs='*', default=[], help='OWPostProcessor overrides for --rank unk, e.g. unk_r_gamma=0.5')
    args = ap.parse_args()
    vd = {'total_batch_size': args.bs, 'shuffle': False,
          'dataset': {'ann_file': args.ann, 'split': 'test'}}
    pp = {k: float(v) for k, v in (a.split('=') for a in args.pp)}
    cfg = YAMLConfig(args.config, val_dataloader=vd, **({'OWPostProcessor': pp} if pp else {}))
    gt = json.load(open(args.ann))
    res, K = predict(cfg, args.resume, gt, max(args.ks), args.num_known, args.rank, args.area_exp)
    print(f'== {args.resume} on {args.ann} (rank {args.rank}, area^{args.area_exp:g})', flush=True)
    if args.dump:
        torch.save(res, args.dump)
        print(f'  saved {args.dump}', flush=True)
        return
    if any(a['category_id'] >= K for a in gt['annotations']):
        score(res, gt, K, args.ks)
    else:
        assert args.out_prefix, '--out-prefix is required for a file without unknown GT'
        write(res, gt, K, args.ks, args.out_prefix, args.resume)


if __name__ == '__main__':
    main()
