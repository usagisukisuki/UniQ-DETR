"""Inference-only unknown-score variants (docs/NEXT_STEPS.md, "A/B/C"), split-half evaluation.

Per decoder query q (known score stays p(c|q)):
  base      o * sigmoid(u)                         current C1 (learned unknown logit)
  A  cons   flip consistency: IoU of q's box with the best box predicted on the h-flipped image
  B  free   max_c p(c|q) if q is not itself a confident known detection (max p < tau) and does not
            overlap one (IoU < 0.5 with any query of max p >= tau), else 0  ("known-like but not known")
  C         products of the above
Each variant is ranked together with the known scores after multiplying by unk_weight w; w is chosen
on the 'tune' half (Known AP50 >= E0 - 3 = 61.5, max U-Recall) and reported on the 'report' half
(same split as tools/sweep_unk.py).

  uv run python tools/sweep_unk_signals.py -c configs/e2_obj.yml -r outputs/e2_obj/best.pth
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
import torchvision  # noqa: E402
from tqdm import tqdm  # noqa: E402

import owod.register  # noqa: F401,E402
from src.core import YAMLConfig  # noqa: E402

from owod.eval.owod_eval import OWODEvaluator  # noqa: E402
from sweep_unk import split_gt  # noqa: E402

WEIGHTS = [1.0, 0.7, 0.5, 0.3, 0.2, 0.1]
KNOWN_FLOOR = 64.51 - 3.0     # E0 Known AP50 minus the Gate's 3pt


def batched_iou(a, b):
    """a [B,Q,4], b [B,R,4] xyxy -> [B,Q,R]"""
    return torch.stack([torchvision.ops.box_iou(x, y) for x, y in zip(a, b)])


def variants(o, su, pmax, cons, boxes):
    out = {'base o*su': o * su, 'A cons': cons, 'A o*su*cons': o * su * cons}
    for tau in (0.3, 0.5):
        conf = pmax >= tau                                           # [B,Q]
        ov = batched_iou(boxes, boxes) >= 0.5                       # [B,Q,Q]
        near_conf = (ov & conf.unsqueeze(1)).any(-1)                 # overlaps a confident known query
        free = (~conf) & (~near_conf)
        b = pmax * free
        out[f'B free(t={tau})'] = b
        out[f'C free(t={tau})*cons'] = b * cons
        out[f'C o*su*cons*free(t={tau})'] = o * su * cons * free
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-c', '--config', required=True)
    ap.add_argument('-r', '--resume', required=True)
    ap.add_argument('--bs', type=int, default=16)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    cfg = YAMLConfig(args.config, val_dataloader={'total_batch_size': args.bs})
    state = torch.load(args.resume, map_location='cpu')
    model = cfg.model
    model.load_state_dict(state['ema']['module'])
    model = model.cuda().eval()
    K = cfg.yaml_cfg['num_classes']

    gt = json.load(open(cfg.yaml_cfg['owod_eval']['gt_json']))
    halves = split_gt(gt)
    half_of = {i['id']: h for h, g in halves.items() for i in g['images']}
    evs = {}

    with torch.no_grad():
        for x, tg in tqdm(cfg.val_dataloader, desc='test'):
            x = x.cuda()
            out, outf = model(x), model(x.flip(-1))
            cxcywh = out['pred_boxes'].float()
            fb = outf['pred_boxes'].float().clone()
            fb[..., 0] = 1 - fb[..., 0]
            b = torchvision.ops.box_convert(cxcywh, 'cxcywh', 'xyxy')
            cons = batched_iou(b, torchvision.ops.box_convert(fb, 'cxcywh', 'xyxy')).max(-1).values
            p = out['pred_logits'].float().sigmoid()
            o = out['pred_objectness'].float()
            su = out['pred_unk_logits'].float().sigmoid()
            sizes = torch.stack([t['orig_size'] for t in tg]).cuda()
            boxes = b * sizes.repeat(1, 2).unsqueeze(1)
            ids = [t['image_id'].item() for t in tg]
            for name, unk in variants(o, su, p.max(-1).values, cons, b).items():
                for w in WEIGHTS:
                    scores = torch.cat([p, (w * unk).unsqueeze(-1)], dim=-1)
                    top, idx = torch.topk(scores.flatten(1), 100, dim=-1)
                    labels, q = idx % (K + 1), idx // (K + 1)
                    bx = boxes.gather(1, q.unsqueeze(-1).repeat(1, 1, 4))
                    for j, i in enumerate(ids):
                        key = (name, w, half_of[i])
                        if key not in evs:
                            evs[key] = OWODEvaluator(halves[key[2]], num_known=K)
                        evs[key].update({i: dict(labels=labels[j], boxes=bx[j], scores=top[j])})

    res = {}
    for (name, w, h), ev in evs.items():
        ev.accumulate()
        st = ev.summarize(verbose=False)
        res.setdefault(name, {}).setdefault(w, {})[h] = {k: st[k] for k in ('K_AP50', 'U_R50', 'WI', 'A_OSE', 'U_AP50')}
    print(f'{"variant":28s} {"w":>4s} | tune K   U    | report K   U     WI     A-OSE')
    best = {}
    for name, by_w in res.items():
        for w in WEIGHTS:
            t, r = by_w[w]['tune'], by_w[w]['report']
            print(f'{name:28s} {w:4.2f} | {t["K_AP50"]:5.2f} {t["U_R50"]:5.2f} | {r["K_AP50"]:5.2f} {r["U_R50"]:5.2f} '
                  f'{r["WI"]:.4f} {r["A_OSE"]:6.0f}')
        ok = [w for w in WEIGHTS if by_w[w]['tune']['K_AP50'] >= KNOWN_FLOOR]
        if ok:
            wb = max(ok, key=lambda w: by_w[w]['tune']['U_R50'])
            best[name] = (wb, by_w[wb]['report'])
    print('\nselected on tune (Known >= %.1f, max U-Recall) -> report half:' % KNOWN_FLOOR)
    for name, (w, r) in best.items():
        print(f'  {name:28s} w={w:4.2f} | Known {r["K_AP50"]:5.2f}  U-Recall {r["U_R50"]:5.2f}  WI {r["WI"]:.4f}  '
              f'A-OSE {r["A_OSE"]:.0f}  U-AP50 {r["U_AP50"]:.2f}')
    out = args.out or os.path.join(os.path.dirname(args.resume), 'sweep_unk_signals.json')
    json.dump({k: {str(w): v for w, v in d.items()} for k, d in res.items()}, open(out, 'w'), indent=1)
    print('saved', out)


if __name__ == '__main__':
    main()
