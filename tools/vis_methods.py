"""Side-by-side detections of several open-world models on sample test images (docs/BASELINES_RESULTS.md).

One figure per image: GT followed by one panel per method, each with its own inference setting (the unk_weight
selected on the tune half). Display filter, identical for every method: known detections with score >= --known-thr
(at most --known-max per image), unknown detections NMS-ed (IoU 0.5), those overlapping a shown known box (IoU >= 0.5)
dropped, top --unk-topk kept. Unknown GT / predictions are labelled 'unknown' (GT also shows the COCO class).

  uv run python tools/vis_methods.py
"""
import argparse
import json
import os
import pickle
import random
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
import torchvision  # noqa: E402
from PIL import Image  # noqa: E402

import owod.register  # noqa: F401,E402
from src.core import YAMLConfig  # noqa: E402

from owod.data.classes import ALL_CLASSES, UNKNOWN_NAME  # noqa: E402

# name, config, checkpoint, unk_weight selected on the tune half (docs/BASELINES_RESULTS.md)
METHODS = [
    ('M0 (ours)', 'configs/f1_quality_50ep.yml', 'outputs/f1_quality_50ep/best.pth', 1.5),
    ('PROB-RT', 'configs/b_prob_rt.yml', 'outputs/e2_obj/best.pth', 0.6),
    ('OW-DETR-RT', 'configs/baselines/owdetr_rt.yml', 'outputs/b_owdetr_rt/best.pth', 32.0),
    ('CAT-RT', 'configs/baselines/cat_rt.yml', 'outputs/b_cat_rt/best.pth', 2.0),
    ('RandBox-RT', 'configs/baselines/randbox_rt.yml', 'outputs/b_randbox_rt/best.pth', 128.0),
    ('OrthogonalDet-RT', 'configs/baselines/orthogonaldet_rt.yml', 'outputs/b_orthogonaldet_rt/best.pth', 8.0),
]
KNOWN_COLOR, UNK_COLOR = '#1f77b4', '#d62728'


def select(d, K, known_thr, known_max, unk_topk, iou_thr=0.5):
    labels, boxes, scores = d['labels'].cpu(), d['boxes'].float().cpu(), d['scores'].float().cpu()
    km = (labels < K) & (scores >= known_thr)
    kb, kl, ks = boxes[km][:known_max], labels[km][:known_max], scores[km][:known_max]
    um = labels == K
    ub, us = boxes[um], scores[um]
    keep = torchvision.ops.nms(ub, us, iou_thr)
    ub, us = ub[keep], us[keep]
    if len(kb) and len(ub):
        free = torchvision.ops.box_iou(ub, kb).max(1).values < iou_thr
        ub, us = ub[free], us[free]
    ub, us = ub[:unk_topk], us[:unk_topk]
    return ([(b, ALL_CLASSES[int(l)], s, KNOWN_COLOR) for b, l, s in zip(kb.tolist(), kl.tolist(), ks.tolist())] +
            [(b, UNKNOWN_NAME, s, UNK_COLOR) for b, s in zip(ub.tolist(), us.tolist())])


def draw(ax, img, items, title):
    W, H = img.size
    ax.imshow(img)
    ax.set_xlim(0, W); ax.set_ylim(H, 0)
    ax.set_title(title, fontsize=11)
    ax.axis('off')
    for (x1, y1, x2, y2), name, s, color in items:
        x1, x2 = max(0, min(x1, W)), max(0, min(x2, W))
        y1, y2 = max(0, min(y1, H)), max(0, min(y2, H))
        if x2 - x1 < 1 or y2 - y1 < 1:
            continue
        ax.add_patch(mpatches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, linewidth=2))
        txt = name if s is None else f'{name} {s:.2f}'
        ax.text(x1, y1, txt, fontsize=6.5, color='white', va='top' if y1 < 0.05 * H else 'bottom', clip_on=True,
                bbox=dict(facecolor=color, alpha=0.8, pad=1, edgecolor='none'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--num-images', type=int, default=12)
    ap.add_argument('--image-ids', default=None)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--known-thr', type=float, default=0.25)
    ap.add_argument('--known-max', type=int, default=10)
    ap.add_argument('--unk-topk', type=int, default=5)
    ap.add_argument('--out-dir', default='outputs/vis_methods')
    ap.add_argument('--_method', type=int, default=None, help=argparse.SUPPRESS)   # worker: one method per process
    ap.add_argument('--_ids', default=None, help=argparse.SUPPRESS)
    ap.add_argument('--_out', default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args._method is not None:
        # the RT-DETR config registry is process-global, so each method is run in its own process
        name, cfg_path, ckpt, w = METHODS[args._method]
        cfg = YAMLConfig(cfg_path, OWPostProcessor={'unk_weight': w})
        model = cfg.model
        model.load_state_dict(torch.load(ckpt, map_location='cpu')['ema']['module'])
        model = model.cuda().eval()
        pp, ds = cfg.postprocessor, cfg.val_dataloader.dataset
        id2idx = {i: k for k, i in enumerate(ds.ids)}
        res = {}
        with torch.no_grad():
            for iid in [int(i) for i in args._ids.split(',')]:
                x, tg = ds[id2idx[iid]]
                d = pp(model(x.unsqueeze(0).cuda()), tg['orig_size'].unsqueeze(0).cuda())[0]
                res[iid] = select(d, 20, args.known_thr, args.known_max, args.unk_topk)
        pickle.dump(res, open(args._out, 'wb'))
        return

    gt = json.load(open('data/mowodb/test.json'))
    K = 20
    anns = {}
    for a in gt['annotations']:
        anns.setdefault(a['image_id'], []).append(a)
    info = {i['id']: i for i in gt['images']}
    if args.image_ids:
        img_ids = [int(i) for i in args.image_ids.split(',')]
    else:   # same sampling as tools/vis_e1.py
        cand = [i['id'] for i in gt['images']
                if any(a['category_id'] >= K for a in anns.get(i['id'], []))
                and any(a['category_id'] < K for a in anns.get(i['id'], []))]
        img_ids = sorted(random.Random(args.seed).sample(cand, min(args.num_images, len(cand))))

    os.makedirs(args.out_dir, exist_ok=True)
    preds = {}   # method -> {img_id: items}
    for m, (name, _, _, _) in enumerate(METHODS):
        tmp = os.path.join(args.out_dir, f'.preds_{m}.pkl')
        subprocess.run([sys.executable, __file__, '--_method', str(m), '--_ids', ','.join(map(str, img_ids)),
                        '--_out', tmp, '--known-thr', str(args.known_thr), '--known-max', str(args.known_max),
                        '--unk-topk', str(args.unk_topk)], check=True, stdout=subprocess.DEVNULL)
        preds[name] = pickle.load(open(tmp, 'rb'))
        os.remove(tmp)
        print('done', name, flush=True)

    for iid in img_ids:
        img = Image.open(info[iid]['file_name']).convert('RGB')
        gtb = []
        for a in anns.get(iid, []):
            x0, y0, w, h = a['bbox']
            c = a['category_id']
            unk = c >= K
            gtb.append(([x0, y0, x0 + w, y0 + h], f'{UNKNOWN_NAME} ({ALL_CLASSES[c]})' if unk else ALL_CLASSES[c], None,
                        UNK_COLOR if unk else KNOWN_COLOR))
        n = 1 + len(METHODS)
        cols = 4
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, (4.6 * img.height / img.width + 0.5) * rows))
        axes = axes.flatten()
        draw(axes[0], img, gtb, 'GT')
        for ax, (name, _, _, w) in zip(axes[1:], METHODS):
            draw(ax, img, preds[name][iid], f'{name} (w={w:g})')
        for ax in axes[n:]:
            ax.axis('off')
        fig.suptitle(f'{os.path.basename(info[iid]["file_name"])} (id={iid})   known: score>={args.known_thr} (max {args.known_max})'
                     f'   unknown: top-{args.unk_topk} after NMS, not overlapping shown known', fontsize=10)
        fig.tight_layout()
        path = os.path.join(args.out_dir, f'{iid:06d}.png')
        fig.savefig(path, dpi=120, bbox_inches='tight')
        plt.close(fig)
        print('saved', path)


if __name__ == '__main__':
    main()
