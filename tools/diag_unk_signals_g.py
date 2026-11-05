"""G4-2 (analysis/G4_PLAN.md): which per-query signals separate unknown objects from the rest, for G-family checkpoints.

Among the final-layer decoder queries that do not overlap a known GT (IoU < 0.5), label = box IoU >= 0.5 with an
unknown GT. Reports AUROC and the hit rate of the per-image top-k for each signal and for the unknown score
o * sigmoid(u) * q * sigmoid(r)^gamma multiplied by (max known p)^beta and p_unk^delta. COCO-sourced test images.

  uv run python tools/diag_unk_signals_g.py -c configs/m0_primary_owdetr.yml -r outputs/m0_primary_owdetr/best.pth
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
from tools.diag_c3c4_signals import auroc, iou  # noqa: E402

NUM_KNOWN = 20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-c', '--config', required=True)
    ap.add_argument('-r', '--resume', required=True)
    ap.add_argument('--k', type=int, default=5)
    ap.add_argument('--gamma', type=float, default=0.5)
    ap.add_argument('--bs', type=int, default=16)
    args = ap.parse_args()

    cfg = YAMLConfig(args.config, val_dataloader={'total_batch_size': args.bs, 'shuffle': False})
    model = cfg.model
    model.load_state_dict(torch.load(args.resume, map_location='cpu')['ema']['module'])
    model = model.cuda().eval()
    gt = json.load(open(cfg.yaml_cfg['owod_eval']['gt_json']))
    anns = defaultdict(list)
    for a in gt['annotations']:
        anns[a['image_id']].append(a)
    src = {i['id']: i['source'] for i in gt['images']}

    sig, lab, n_img = defaultdict(list), [], 0
    with torch.no_grad():
        for x, tg in cfg.val_dataloader:
            keep = [j for j, t in enumerate(tg) if src[t['image_id'].item()] == 'coco']
            if not keep:
                continue
            out = model(x.cuda())
            p = out['pred_logits'].float().sigmoid().cpu()
            for j in keep:
                iid = tg[j]['image_id'].item()
                w, h = tg[j]['orig_size'].tolist()
                scale = torch.tensor([w, h, w, h], dtype=torch.float32)
                box = lambda L: torch.tensor([[a['bbox'][0], a['bbox'][1], a['bbox'][0] + a['bbox'][2],
                                               a['bbox'][1] + a['bbox'][3]] for a in L], dtype=torch.float32).reshape(-1, 4)
                gk = box([a for a in anns[iid] if a['category_id'] < NUM_KNOWN and not a['difficult']])
                gu = box([a for a in anns[iid] if a['category_id'] >= NUM_KNOWN])
                db = torchvision.ops.box_convert(out['pred_boxes'][j].float().cpu(), 'cxcywh', 'xyxy') * scale
                free = (iou(db, gk).max(1).values < 0.5) if len(gk) else torch.ones(len(db), dtype=torch.bool)
                if free.sum() == 0:
                    continue
                y = (iou(db[free], gu).max(1).values >= 0.5) if len(gu) else torch.zeros(int(free.sum()), dtype=torch.bool)
                g = lambda k: out[k][j].float().cpu()[free]
                s = {'o': g('pred_objectness'), 'sig(u)': g('pred_unk_logits').sigmoid(),
                     'q': g('pred_quality_logits').sigmoid(), 'sig(r)': g('pred_primary_logits').sigmoid(),
                     'max p': p[j, :, :NUM_KNOWN].max(-1).values[free]}
                if p.shape[-1] > NUM_KNOWN:
                    s['p_unk'] = p[j, :, NUM_KNOWN][free]
                s['energy'] = torch.logsumexp(out['pred_logits'][j, :, :NUM_KNOWN].float().cpu(), -1)[free]
                s['log area'] = ((db[:, 2] - db[:, 0]) * (db[:, 3] - db[:, 1]) / (w * h)).clamp(min=1e-6).log()[free]
                for k, v in s.items():
                    sig[k].append(v)
                lab.append(y)
                n_img += 1

    y = torch.cat(lab)
    S = {k: torch.cat(v).double() for k, v in sig.items()}
    base = S['o'] * S['sig(u)'] * S['q'] * S['sig(r)'] ** args.gamma
    S[f'G3 score (gamma {args.gamma})'] = base
    for b in (0.5, 1, 2):
        S[f'G3 * maxp^{b}'] = base * S['max p'] ** b
    if 'p_unk' in S:
        for d in (0.5, 1, 2):
            S[f'G3 * p_unk^{d}'] = base * S['p_unk'] ** d
        for b, d in ((0.5, 0.5), (1, 1), (1, 0.5), (0.5, 1)):
            S[f'G3 * maxp^{b} * p_unk^{d}'] = base * S['max p'] ** b * S['p_unk'] ** d
    # per-image top-k hit rate needs the per-image split
    sizes = [len(t) for t in lab]
    print(f'images {n_img} | non-known queries {len(y)} | on an unknown object {100 * y.float().mean():.2f}%')
    for k, v in S.items():
        hit, n = 0., 0
        for vv, yy in zip(v.split(sizes), lab):
            t = vv.topk(min(args.k, len(vv))).indices
            hit += float(yy[t].sum()); n += len(t)
        print(f'  {k:28s} AUROC {auroc(v, y):.3f} | top-{args.k}/image on unknown {100 * hit / n:5.1f}%')


if __name__ == '__main__':
    main()
