"""Sweep the known/unknown balance of the top-100 (OWPostProcessor.unk_weight / unk_slots).

M-OWODB has no validation split, so the test set is split in two halves by image name parity
(duplicate listings stay together): settings are chosen on the 'tune' half and reported on the
'report' half. One forward pass; every setting is scored from the same raw outputs.

  uv run python tools/sweep_unk.py -c configs/e2_obj.yml -r outputs/e2_obj/best.pth
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
from tqdm import tqdm  # noqa: E402

import owod.register  # noqa: F401,E402
from src.core import YAMLConfig  # noqa: E402

from owod.eval.owod_eval import OWODEvaluator  # noqa: E402
from owod.zoo.ow_postprocessor import OWPostProcessor  # noqa: E402

WEIGHTS = [1.0, 0.7, 0.5, 0.3, 0.2, 0.1, 0.05]
SLOTS = [None, 50, 30, 20, 10, 5]


def split_gt(gt):
    halves = {}
    for name, par in (('tune', 0), ('report', 1)):
        imgs = [i for i in gt['images'] if int(i['owod_name']) % 2 == par]
        ids = {i['id'] for i in imgs}
        halves[name] = dict(info=gt['info'], categories=gt['categories'], images=imgs,
                            annotations=[a for a in gt['annotations'] if a['image_id'] in ids])
    return halves


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-c', '--config', required=True)
    ap.add_argument('-r', '--resume', required=True)
    ap.add_argument('--bs', type=int, default=32)
    ap.add_argument('--out', default=None)
    ap.add_argument('--unk-score', default=None, help='override OWPostProcessor.unk_score')
    args = ap.parse_args()

    cfg = YAMLConfig(args.config, val_dataloader={'total_batch_size': args.bs})
    device = torch.device('cuda')
    state = torch.load(args.resume, map_location='cpu')
    model = cfg.model
    model.load_state_dict(state['ema']['module'] if 'ema' in state else state['model'])
    model = model.to(device).eval()

    base = cfg.yaml_cfg['OWPostProcessor']
    settings = [(w, None) for w in WEIGHTS] + [(1.0, n) for n in SLOTS if n is not None] \
        + [(w, n) for w in (0.5, 0.3) for n in (30, 20)]
    pps = {s: OWPostProcessor(num_classes=cfg.yaml_cfg['num_classes'], num_top_queries=100,
                              unk_score=args.unk_score or base['unk_score'], known_score=base['known_score'],
                              unk_weight=s[0], unk_slots=s[1]) for s in settings}
    gt = json.load(open(cfg.yaml_cfg['owod_eval']['gt_json']))
    halves = split_gt(gt)
    K = cfg.yaml_cfg['owod_eval']['num_known']
    evs = {(s, h): OWODEvaluator(halves[h], num_known=K) for s in settings for h in halves}
    half_of = {i['id']: h for h, g in halves.items() for i in g['images']}

    with torch.no_grad():
        for x, tg in tqdm(cfg.val_dataloader, desc='test'):
            out = model(x.to(device))
            sizes = torch.stack([t['orig_size'] for t in tg]).to(device)
            ids = [t['image_id'].item() for t in tg]
            for s, pp in pps.items():
                for i, r in zip(ids, pp(out, sizes)):
                    evs[(s, half_of[i])].update({i: r})

    rows = []
    for s in settings:
        row = {'unk_weight': s[0], 'unk_slots': s[1]}
        for h in halves:
            evs[(s, h)].accumulate()
            st = evs[(s, h)].summarize(verbose=False)
            row.update({f'{h}_{k}': st[k] for k in ('K_AP50', 'U_R50', 'WI', 'A_OSE', 'U_AP50')})
        rows.append(row)
        print('w={unk_weight:<5} slots={unk_slots!s:<5} | tune K {tune_K_AP50:6.2f} U {tune_U_R50:6.2f} WI {tune_WI:.4f} '
              '| report K {report_K_AP50:6.2f} U {report_U_R50:6.2f} WI {report_WI:.4f} A-OSE {report_A_OSE:.0f}'.format(**row),
              flush=True)
    out = args.out or os.path.join(os.path.dirname(args.resume), 'sweep_unk.json')
    json.dump(rows, open(out, 'w'), indent=1)
    print('saved', out)


if __name__ == '__main__':
    main()
