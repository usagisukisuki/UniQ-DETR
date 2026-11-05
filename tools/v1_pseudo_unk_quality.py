"""V1 (docs/NEXT_STEPS.md): quality of pseudo-unknown labels made from a teacher's own predictions.

Probe set: data/v1_probe/coco_train_probe3000.json = 3,000 COCO train2017 images containing >=1 VOC
class (not used in Task-1 training or testing). Known GT (category < 20) plays the role of the
training labels; unknown GT (category >= 20) is used ONLY to score the pseudo labels.

Per image, candidates = teacher decoder boxes that do not overlap known GT (IoU < 0.5), deduplicated
by NMS (IoU 0.5) on the rule's score. A rule keeps the top-k candidates by its score.
  precision = kept boxes with IoU >= 0.5 to an unknown GT / kept boxes
  recall    = unknown GT hit by >= 1 kept box / all unknown GT

Scores: maxp = max_c p(c), su = sigmoid(u), o, area (fraction of image), cons = IoU with the best
matching box predicted on the horizontally flipped image (flip consistency), and products / an
area >= 1% gate.

  uv run python tools/v1_pseudo_unk_quality.py -c configs/e2_obj.yml \
      -r outputs/e2_pilot_bn/last.pth outputs/e2_obj/checkpoint0009.pth ...
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

KS = [1, 3, 5, 10]


def iou(a, b):
    return torchvision.ops.box_iou(a, b) if len(a) and len(b) else torch.zeros(len(a), len(b))


def rules(c):
    big = (c['area'] >= 0.01).float()
    return {
        'maxp': c['maxp'], 'su': c['su'], 'o': c['o'], 'area': c['area'], 'cons': c['cons'],
        'maxp*cons': c['maxp'] * c['cons'], 'area*cons': c['area'] * c['cons'],
        'maxp|area>=1%': c['maxp'] * big, 'maxp*cons|area>=1%': c['maxp'] * c['cons'] * big,
        'o*su': c['o'] * c['su'],
    }


def evaluate(cfg, ckpt, ann, max_images):
    state = torch.load(ckpt, map_location='cpu')
    model = cfg.model
    model.load_state_dict(state['ema']['module'])
    model = model.cuda().eval()
    K = cfg.yaml_cfg['num_classes']
    gt = json.load(open(ann))
    anns = defaultdict(list)
    for a in gt['annotations']:
        anns[a['image_id']].append(a)

    hit = defaultdict(float)        # (rule, k) -> kept boxes on unknown
    kept = defaultdict(float)       # (rule, k) -> kept boxes
    rec = defaultdict(float)        # (rule, k) -> unknown GT recalled
    n_unk, n_cand, n_cand_unk, n_img = 0., 0., 0., 0
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
                box = lambda L: torch.tensor([[a['bbox'][0], a['bbox'][1], a['bbox'][0] + a['bbox'][2],
                                               a['bbox'][1] + a['bbox'][3]] for a in L], dtype=torch.float32).reshape(-1, 4)
                gk = box([a for a in anns[iid] if a['category_id'] < K])
                gu = box([a for a in anns[iid] if a['category_id'] >= K])
                n_unk += len(gu)
                db = xyxy(out['pred_boxes'][j])
                fb = outf['pred_boxes'][j].clone()
                fb[:, 0] = 1 - fb[:, 0]                      # un-flip cx
                fb = xyxy(fb)
                free = (iou(db, gk).max(1).values < 0.5) if len(gk) else torch.ones(len(db), dtype=torch.bool)
                if free.sum() == 0:
                    n_img += 1
                    continue
                b = db[free]
                c = {'maxp': out['pred_logits'][j].float().sigmoid().max(-1).values.cpu()[free],
                     'su': out['pred_unk_logits'][j].float().sigmoid().cpu()[free],
                     'o': out['pred_objectness'][j].float().cpu()[free],
                     'area': ((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) / (w * h)).clamp(min=0),
                     'cons': iou(b, fb).max(1).values}
                on_unk = (iou(b, gu) >= 0.5) if len(gu) else torch.zeros(len(b), 0, dtype=torch.bool)
                n_cand += len(b); n_cand_unk += float(on_unk.any(1).sum()) if len(gu) else 0.
                for name, s in rules(c).items():
                    keep = torchvision.ops.nms(b, s.float() + 1e-9, 0.5)   # sorted by score
                    for k in KS:
                        sel = keep[:k]
                        kept[(name, k)] += len(sel)
                        if len(gu):
                            hit[(name, k)] += float(on_unk[sel].any(1).sum())
                            rec[(name, k)] += float(on_unk[sel].any(0).sum())
                n_img += 1
            if n_img >= max_images:
                break
    res = {'n_img': n_img, 'n_unk': n_unk, 'base_rate': n_cand_unk / max(n_cand, 1), 'rules': {}}
    for (name, k) in kept:
        res['rules'][f'{name}@{k}'] = {'precision': hit[(name, k)] / max(kept[(name, k)], 1),
                                      'recall': rec[(name, k)] / max(n_unk, 1)}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-c', '--config', required=True)
    ap.add_argument('-r', '--resume', nargs='+', required=True)
    ap.add_argument('--ann', default='data/v1_probe/coco_train_probe3000.json')
    ap.add_argument('--max-images', type=int, default=3000)
    ap.add_argument('--bs', type=int, default=16)
    ap.add_argument('--out', default='outputs/v1_pseudo_unk_quality.json')
    args = ap.parse_args()
    vd = {'total_batch_size': args.bs, 'shuffle': False,
          'dataset': {'ann_file': args.ann, 'split': 'test'}}
    allres = {}
    for ck in args.resume:
        cfg = YAMLConfig(args.config, val_dataloader=vd)
        r = evaluate(cfg, ck, args.ann, args.max_images)
        allres[ck] = r
        print(f'== {ck} | images {r["n_img"]} unknown GT {r["n_unk"]:.0f} | '
              f'candidates on unknown (base rate) {100 * r["base_rate"]:.1f}%', flush=True)
        for name in rules({k: torch.zeros(1) for k in ('maxp', 'su', 'o', 'area', 'cons')}):
            line = '  '.join(f'@{k} P {100 * r["rules"][f"{name}@{k}"]["precision"]:5.1f} R {100 * r["rules"][f"{name}@{k}"]["recall"]:5.1f}'
                             for k in KS)
            print(f'  {name:20s} {line}', flush=True)
        json.dump(allres, open(args.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
