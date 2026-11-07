"""Side-by-side G4-6 vs G4-8 (UniQ-DETR inference) on COCO test images, showing where A-OSE comes from
(analysis/G4_PLAN.md, A-OSE analysis).

Each panel draws, from the raw top-100 (no NMS, no filtering other than below):
  blue    known detections with score >= --known-thr
  orange  OSE detections: known-class detections overlapping an unknown GT (IoU > 0.5), at any score (= A-OSE)
  red     for each unknown GT, the best-scoring unknown detection hitting it (IoU >= 0.5)
Images: the --top ones with the largest OSE increase from the first to the second model, plus --random others.

  uv run python tools/vis_aose.py
"""
import argparse
import json
import os
import pickle
import random
import subprocess
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
import torchvision  # noqa: E402
from PIL import Image  # noqa: E402

from owod.data.classes import ALL_CLASSES, UNKNOWN_NAME  # noqa: E402

# name, checkpoint, decoder queries (configs/uniq_detr.yml, w 12)
MODELS = [
    ('G4-6 (900 queries)', 'outputs/g4_6/best.pth', 900),
    ('G4-8 (600 queries) = UniQ-DETR', 'outputs/g4_8/best.pth', 600),
]
K = 20
KNOWN_COLOR, OSE_COLOR, UNK_COLOR = '#1f77b4', '#ff7f0e', '#d62728'


def run_model(ckpt, nq, out):
    import owod.register  # noqa: F401
    from src.core import YAMLConfig
    cfg = YAMLConfig('configs/uniq_detr.yml', OWRTDETRTransformerv2={'num_queries': nq},
                     val_dataloader={'total_batch_size': 16})
    model = cfg.model
    model.load_state_dict(torch.load(ckpt, map_location='cpu')['ema']['module'])
    model = model.cuda().eval()
    pp = cfg.postprocessor
    src = {i['id']: i['source'] for i in json.load(open(cfg.yaml_cfg['owod_eval']['gt_json']))['images']}
    res = {}
    with torch.no_grad():
        for x, tg in cfg.val_dataloader:
            keep = [j for j, t in enumerate(tg) if src[t['image_id'].item()] == 'coco']
            if not keep:
                continue
            sizes = torch.stack([t['orig_size'] for t in tg]).cuda()
            dets = pp(model(x.cuda()), sizes)
            for j in keep:
                d = dets[j]
                res[tg[j]['image_id'].item()] = (d['labels'].cpu(), d['boxes'].float().cpu(), d['scores'].float().cpu())
    pickle.dump(res, open(out, 'wb'))


def boxes_of(anns):
    return torch.tensor([[a['bbox'][0], a['bbox'][1], a['bbox'][0] + a['bbox'][2], a['bbox'][1] + a['bbox'][3]]
                         for a in anns], dtype=torch.float32).reshape(-1, 4)


def classify(det, unk_anns, known_thr):
    """-> items to draw, number of OSE detections, unknown GT hit, known detections in the top-100."""
    lab, bx, sc = det
    isk = lab < K
    gu = boxes_of(unk_anns)
    ov = torchvision.ops.box_iou(bx, gu) if len(gu) else torch.zeros(len(bx), 0)
    ose = isk & (ov.max(1).values > 0.5) if len(gu) else torch.zeros_like(isk)
    items = [(b, ALL_CLASSES[int(l)], s, KNOWN_COLOR, '-') for b, l, s in
             zip(bx[isk & ~ose & (sc >= known_thr)].tolist(), lab[isk & ~ose & (sc >= known_thr)].tolist(),
                 sc[isk & ~ose & (sc >= known_thr)].tolist())]
    items += [(b, ALL_CLASSES[int(l)], s, OSE_COLOR, '--') for b, l, s in
              zip(bx[ose].tolist(), lab[ose].tolist(), sc[ose].tolist())]
    hit = 0
    for g in range(len(gu)):
        m = ~isk & (ov[:, g] >= 0.5)
        if m.any():
            i = torch.where(m)[0][sc[m].argmax()]
            items.append((bx[i].tolist(), UNKNOWN_NAME, sc[i].item(), UNK_COLOR, '-'))
            hit += 1
    return items, int(ose.sum()), hit, int(isk.sum())


def draw(ax, img, items, title):
    W, H = img.size
    ax.imshow(img)
    ax.set_xlim(0, W); ax.set_ylim(H, 0)
    ax.set_title(title, fontsize=10)
    ax.axis('off')
    for (x1, y1, x2, y2), name, s, color, ls in items:
        x1, x2 = max(0, min(x1, W)), max(0, min(x2, W))
        y1, y2 = max(0, min(y1, H)), max(0, min(y2, H))
        if x2 - x1 < 1 or y2 - y1 < 1:
            continue
        ax.add_patch(mpatches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, linewidth=1.8,
                                        linestyle=ls))
        txt = name if s is None else f'{name} {s:.2f}'
        ax.text(x1, y1, txt, fontsize=6, color='white', va='top' if y1 < 0.05 * H else 'bottom', clip_on=True,
                bbox=dict(facecolor=color, alpha=0.8, pad=1, edgecolor='none'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--top', type=int, default=8)
    ap.add_argument('--random', type=int, default=4)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--known-thr', type=float, default=0.3)
    ap.add_argument('--out-dir', default='outputs/vis_aose')
    ap.add_argument('--_model', type=int, default=None, help=argparse.SUPPRESS)   # worker: one model per process
    ap.add_argument('--_out', default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args._model is not None:
        _, ckpt, nq = MODELS[args._model]
        run_model(ckpt, nq, args._out)
        return

    os.makedirs(args.out_dir, exist_ok=True)
    preds = []
    for m in range(len(MODELS)):
        tmp = os.path.join(args.out_dir, f'.preds_{m}.pkl')
        if not os.path.exists(tmp):
            subprocess.run([sys.executable, __file__, '--_model', str(m), '--_out', tmp], check=True,
                           stdout=subprocess.DEVNULL)
        preds.append(pickle.load(open(tmp, 'rb')))
        print('done', MODELS[m][0], flush=True)

    gt = json.load(open('data/mowodb/test.json'))
    anns = defaultdict(list)
    for a in gt['annotations']:
        anns[a['image_id']].append(a)
    info = {i['id']: i for i in gt['images']}
    ids = [i for i in preds[0] if any(a['category_id'] >= K for a in anns[i])]
    stats = {i: [classify(p[i], [a for a in anns[i] if a['category_id'] >= K], args.known_thr) for p in preds]
             for i in ids}
    inc = sorted(ids, key=lambda i: stats[i][1][1] - stats[i][0][1], reverse=True)
    chosen = inc[:args.top]
    rest = [i for i in ids if i not in chosen]
    chosen += sorted(random.Random(args.seed).sample(rest, args.random))
    tot = [[sum(stats[i][m][k] for i in ids) for k in (1, 2, 3)] for m in range(len(MODELS))]
    for (name, *_), (o, h, k) in zip(MODELS, tot):
        print(f'{name}: OSE {o}, unknown GT hit {h}, known dets in top-100 {k / len(ids):.1f}/img ({len(ids)} images)')

    for rank, iid in enumerate(chosen):
        img = Image.open(info[iid]['file_name']).convert('RGB')
        gtb = []
        for a in anns[iid]:
            x0, y0, w, h = a['bbox']
            c = a['category_id']
            unk = c >= K
            gtb.append(([x0, y0, x0 + w, y0 + h], f'{UNKNOWN_NAME} ({ALL_CLASSES[c]})' if unk else ALL_CLASSES[c], None,
                        UNK_COLOR if unk else KNOWN_COLOR, '-'))
        fig, axes = plt.subplots(1, 1 + len(MODELS), figsize=(5.2 * (1 + len(MODELS)), 5.2 * img.height / img.width + 0.8))
        draw(axes[0], img, gtb, 'GT')
        for ax, (name, *_), (items, o, h, k) in zip(axes[1:], MODELS, stats[iid]):
            draw(ax, img, items, f'{name}\nOSE {o} | unknown GT hit {h} | known dets in top-100 {k}')
        kind = 'largest OSE increase' if rank < args.top else 'random'
        fig.suptitle(f'{os.path.basename(info[iid]["file_name"])} (id={iid}, {kind})   blue: known score>={args.known_thr}'
                     f'   orange dashed: known on an unknown GT (A-OSE, any score)   red: best unknown det per unknown GT',
                     fontsize=9)
        fig.tight_layout()
        path = os.path.join(args.out_dir, f'{rank:02d}_{iid:06d}.png')
        fig.savefig(path, dpi=110, bbox_inches='tight')
        plt.close(fig)
        print('saved', path)


if __name__ == '__main__':
    main()
