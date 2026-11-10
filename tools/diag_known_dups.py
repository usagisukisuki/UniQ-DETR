"""Why does the closed-set Known AP50 drop? Duplicate and error breakdown of the known detections (G4-9 vs G4-8,
analysis/G4_PLAN.md).

Closed-set scoring (p(c) only, no unknown), top-100 per image, all test images. Detections with score >= --thr are
taken in descending score order (VOC-style greedy assignment, IoU > 0.5) and split into
  tp        first detection of a known GT of its class
  dup       same class, IoU > 0.5 to a known GT that is already taken (a duplicate the one-to-one head should not emit)
  cls       IoU > 0.5 to a known GT of another class
  unk       IoU > 0.5 to an unknown GT
  bg        everything else
Also, per known GT: recall over the whole top-100, and the score ratio of the best duplicate to the TP.

  uv run python tools/diag_known_dups.py -c configs/g4_8.yml -r outputs/g4_8/best.pth
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
    ap.add_argument('--thr', type=float, nargs='+', default=[0.05, 0.3])
    args = ap.parse_args()
    cfg = YAMLConfig(args.config, OWPostProcessor={'unk_score': 'none', 'known_score': 'cls'},
                     val_dataloader={'total_batch_size': 16})
    model = cfg.model
    model.load_state_dict(torch.load(args.resume, map_location='cpu')['ema']['module'])
    model = model.cuda().eval()
    pp, K = cfg.postprocessor, cfg.yaml_cfg['owod_eval']['num_known']
    gt = json.load(open(cfg.yaml_cfg['owod_eval']['gt_json']))
    anns = defaultdict(list)
    for a in gt['annotations']:
        anns[a['image_id']].append(a)
    box = lambda L: torch.tensor([[a['bbox'][0], a['bbox'][1], a['bbox'][0] + a['bbox'][2], a['bbox'][1] + a['bbox'][3]]
                                  for a in L], dtype=torch.float32).reshape(-1, 4)
    cnt = {t: defaultdict(int) for t in args.thr}
    n_known, hit_any, ratios = 0, 0, []
    with torch.no_grad():
        for x, tg in cfg.val_dataloader:
            sizes = torch.stack([t['orig_size'] for t in tg]).cuda()
            dets = pp(model(x.cuda()), sizes)
            for j, t in enumerate(tg):
                A = anns[t['image_id'].item()]
                kA = [a for a in A if a['category_id'] < K and not a.get('difficult', 0)]
                uA = [a for a in A if a['category_id'] >= K]
                gk, gu = box(kA), box(uA)
                kc = torch.tensor([a['category_id'] for a in kA])
                d = dets[j]
                lab, bx, sc = d['labels'].cpu(), d['boxes'].float().cpu(), d['scores'].float().cpu()
                order = sc.argsort(descending=True)
                lab, bx, sc = lab[order], bx[order], sc[order]
                ok = torchvision.ops.box_iou(bx, gk) if len(gk) else torch.zeros(len(bx), 0)
                ou = torchvision.ops.box_iou(bx, gu) if len(gu) else torch.zeros(len(bx), 0)
                same = ok * (lab[:, None] == kc[None]) if len(gk) else ok
                n_known += len(kA)
                if len(gk):
                    best = same.max(0).values
                    hit_any += int((best > 0.5).sum())
                for thr in args.thr:
                    taken = torch.zeros(len(kA), dtype=torch.bool)
                    tp_score = torch.zeros(len(kA))
                    for i in torch.where(sc >= thr)[0].tolist():
                        if len(kA) and same[i].max() > 0.5:
                            g = int(same[i].argmax())
                            if not taken[g]:
                                taken[g] = True
                                tp_score[g] = sc[i]
                                cnt[thr]['tp'] += 1
                            else:
                                cnt[thr]['dup'] += 1
                                if thr == args.thr[0]:
                                    ratios.append(float(sc[i] / tp_score[g]))
                        elif len(kA) and ok[i].max() > 0.5:
                            cnt[thr]['cls'] += 1
                        elif len(uA) and ou[i].max() > 0.5:
                            cnt[thr]['unk'] += 1
                        else:
                            cnt[thr]['bg'] += 1
    print(f'{os.path.basename(args.config)} {args.resume} | known GT {n_known} | recall in top-100 {100 * hit_any / n_known:.1f}%')
    for thr in args.thr:
        c = cnt[thr]
        tot = sum(c.values())
        print(f'  score >= {thr:<4}: dets {tot} | tp {c["tp"]} (recall {100 * c["tp"] / n_known:.1f}%) | '
              + ' | '.join(f'{k} {c[k]} ({100 * c[k] / tot:.1f}%)' for k in ('dup', 'cls', 'unk', 'bg')))
    r = torch.tensor(ratios)
    if len(r):
        print(f'  duplicates (score >= {args.thr[0]}): {len(r)}, score / TP score median {r.median():.2f}, '
              f'>= 0.5 {100 * (r >= 0.5).float().mean():.0f}%')


if __name__ == '__main__':
    main()
