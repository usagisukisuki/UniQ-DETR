"""Split-half sweep over arbitrary OWPostProcessor settings (generalises tools/sweep_unk.py).

Settings = product of the --grid lists; w (unk_weight) is chosen on the 'tune' half
(Known AP50 >= 61.5, max U-Recall) per setting and reported on the 'report' half.

  uv run python tools/sweep_pp.py -c configs/f1_quality.yml -r outputs/f1_quality/best.pth \
      --grid unk_score=learned_q,learned unk_dup_tau=0.1,0.3 unk_dup_iou=0.5,0.7 unk_weight=1,0.7,0.5,0.3
"""
import argparse, itertools, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch  # noqa: E402
from tqdm import tqdm  # noqa: E402
import owod.register  # noqa: F401,E402
from src.core import YAMLConfig  # noqa: E402
from owod.eval.owod_eval import OWODEvaluator  # noqa: E402
from owod.zoo.ow_postprocessor import OWPostProcessor  # noqa: E402
from sweep_unk import split_gt  # noqa: E402

FLOOR = 64.51 - 3.0


def parse(v):
    if v == 'None':
        return None
    try:
        return float(v)
    except ValueError:
        return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-c', '--config', required=True); ap.add_argument('-r', '--resume', required=True)
    ap.add_argument('--grid', nargs='+', required=True); ap.add_argument('--bs', type=int, default=32)
    ap.add_argument('--out', default=None)
    ap.add_argument('--floor', type=float, default=FLOOR, help='Known AP50 floor on the tune half (G4: 65.22)')
    args = ap.parse_args()
    floor = args.floor
    keys, vals = zip(*[(g.split('=')[0], [parse(v) for v in g.split('=')[1].split(',')]) for g in args.grid])
    settings = [dict(zip(keys, c)) for c in itertools.product(*vals)]
    cfg = YAMLConfig(args.config, val_dataloader={'total_batch_size': args.bs})
    model = cfg.model; model.load_state_dict(torch.load(args.resume, map_location='cpu')['ema']['module'])
    model = model.cuda().eval()
    base = dict(cfg.yaml_cfg['OWPostProcessor'])
    # known classes for evaluation; num_classes may include an 'unknown' class (baselines: 21 or 81)
    K = cfg.yaml_cfg['owod_eval']['num_known']
    pps = [OWPostProcessor(num_classes=cfg.yaml_cfg['num_classes'], **{**base, **s}) for s in settings]
    gt = json.load(open(cfg.yaml_cfg['owod_eval']['gt_json'])); halves = split_gt(gt)
    half_of = {i['id']: h for h, g in halves.items() for i in g['images']}
    evs = [{h: OWODEvaluator(halves[h], num_known=K) for h in halves} for _ in settings]
    with torch.no_grad():
        for x, tg in tqdm(cfg.val_dataloader, desc='test'):
            out = model(x.cuda()); sizes = torch.stack([t['orig_size'] for t in tg]).cuda()
            ids = [t['image_id'].item() for t in tg]
            for pp, ev in zip(pps, evs):
                for i, r in zip(ids, pp(out, sizes)):
                    ev[half_of[i]].update({i: r})
    rows = []
    for s, ev in zip(settings, evs):
        row = dict(s)
        for h in halves:
            ev[h].accumulate(); st = ev[h].summarize(verbose=False)
            row.update({f'{h}_{k}': st[k] for k in ('K_AP50', 'U_R50', 'WI', 'A_OSE', 'U_AP50')})
        rows.append(row)
    fixed = [k for k in keys if k != 'unk_weight']
    print('per setting: tune Known / U | report Known / U / WI / A-OSE')
    for r in rows:
        print('  ' + ' '.join(f'{k}={r[k]}' for k in keys) +
              f" | {r['tune_K_AP50']:5.2f} {r['tune_U_R50']:5.2f} | {r['report_K_AP50']:5.2f} {r['report_U_R50']:5.2f} "
              f"{r['report_WI']:.4f} {r['report_A_OSE']:.0f}")
    print(f'\nselected w on tune (Known >= {floor:.2f}, max U-Recall) -> report:')
    groups = {}
    for r in rows:
        groups.setdefault(tuple(r[k] for k in fixed), []).append(r)
    for g, rs in groups.items():
        ok = [r for r in rs if r['tune_K_AP50'] >= floor]
        if not ok:
            continue
        b = max(ok, key=lambda r: r['tune_U_R50'])
        print('  ' + ' '.join(f'{k}={v}' for k, v in zip(fixed, g)) + f" w={b.get('unk_weight')} | Known {b['report_K_AP50']:5.2f} "
              f"U-Recall {b['report_U_R50']:5.2f} WI {b['report_WI']:.4f} A-OSE {b['report_A_OSE']:.0f}")
    ok = [r for r in rows if r['tune_K_AP50'] >= floor]
    if ok:
        b = max(ok, key=lambda r: r['tune_U_R50'])
        print('overall best on tune: ' + ' '.join(f'{k}={b[k]}' for k in keys) + f" | Known {b['report_K_AP50']:5.2f} "
              f"U-Recall {b['report_U_R50']:5.2f} WI {b['report_WI']:.4f} A-OSE {b['report_A_OSE']:.0f}")
    out = args.out or os.path.join(os.path.dirname(args.resume), 'sweep_pp.json')
    json.dump(rows, open(out, 'w'), indent=1); print('saved', out)


if __name__ == '__main__':
    main()
