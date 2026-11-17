"""UniQ-DETR inference examples for the README (one strip of images, NMS-free raw top-k output).

Display filter only (no NMS, no duplicate removal): known detections with score >= --known-thr, unknown detections with
score >= --unk-thr (one label per identical box; unk_weight-scaled, so labels show class names only). Candidate images contain both known and unknown GT; an image is kept when at least --min-unk-hits
unknown GT objects are found by a shown unknown box (IoU >= 0.5) and no shown box is a duplicate (IoU >= 0.7 with
another shown box) and, with --clean, at least --clean-frac of the shown boxes match a GT object of their label (and >= --min-boxes are shown), then --num images are drawn at random from those (--seed).

  uv run python tools/vis_uniq.py
"""
import argparse
import collections
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
import torchvision  # noqa: E402
from PIL import Image  # noqa: E402

import owod.register  # noqa: F401,E402
from owod.data.classes import ALL_CLASSES, UNKNOWN_NAME  # noqa: E402
from src.core import YAMLConfig  # noqa: E402

KNOWN_COLOR, UNK_COLOR = '#2a78d6', '#e34948'
K = 20


def shown(d, known_thr, unk_thr):
    labels, boxes, scores = d['labels'].cpu(), d['boxes'].float().cpu(), d['scores'].float().cpu()
    m = ((labels < K) & (scores >= known_thr)) | ((labels == K) & (scores >= unk_thr))
    boxes, labels, scores = boxes[m], labels[m], scores[m]
    # top-k is taken over (query, class) pairs, so one query can appear with several labels on the identical box;
    # draw it once, preferring a known label (known first, then by score)
    order = sorted(range(len(boxes)), key=lambda i: (int(labels[i] == K), -float(scores[i])))
    keep = []
    for i in order:
        if all(torchvision.ops.box_iou(boxes[i:i + 1], boxes[j:j + 1]).item() < 0.95 for j in keep):
            keep.append(i)
    keep = torch.tensor(keep, dtype=torch.long)
    return boxes[keep], labels[keep], scores[keep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-c', '--config', default='configs/uniq_detr.yml')
    ap.add_argument('-r', '--resume', default='outputs/g4_8/best.pth')
    ap.add_argument('--unk-weight', type=float, default=12.)
    ap.add_argument('--known-thr', type=float, default=0.5)
    ap.add_argument('--unk-thr', type=float, default=0.5)   # weighted score (w=12), not a probability
    ap.add_argument('--min-unk-hits', type=int, default=2)
    ap.add_argument('--dup-iou', type=float, default=0.7)
    ap.add_argument('--max-boxes', type=int, default=15)
    ap.add_argument('--candidates', type=int, default=3000)
    ap.add_argument('--clean', type=int, default=1, help='keep only images whose shown boxes mostly match a GT object')
    ap.add_argument('--clean-frac', type=float, default=0.8)
    ap.add_argument('--min-boxes', type=int, default=4)
    ap.add_argument('--list', action='store_true', help='print every image id that passes the filter')
    ap.add_argument('--ids', default=None, help='comma-separated image ids to draw (skips the random draw)')
    ap.add_argument('--num', type=int, default=4)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='analysis/figs/vis_uniq.png')
    args = ap.parse_args()

    cfg = YAMLConfig(args.config, OWPostProcessor={'unk_weight': args.unk_weight})
    model = cfg.model
    model.load_state_dict(torch.load(args.resume, map_location='cpu')['ema']['module'])
    model = model.cuda().eval()
    pp, ds = cfg.postprocessor, cfg.val_dataloader.dataset
    id2idx = {i: k for k, i in enumerate(ds.ids)}

    gt = json.load(open('data/mowodb/test.json'))
    anns = {}
    for a in gt['annotations']:
        anns.setdefault(a['image_id'], []).append(a)
    info = {i['id']: i for i in gt['images']}
    cand = [i for i, a in anns.items() if any(x['category_id'] >= K for x in a) and any(x['category_id'] < K for x in a)]
    cand = random.Random(args.seed).sample(sorted(cand), min(args.candidates, len(cand)))

    good, why = [], collections.Counter()
    with torch.no_grad():
        seen = set()
        for iid in cand:
            if info[iid]['file_name'] in seen:   # test.json lists some COCO images twice
                continue
            seen.add(info[iid]['file_name'])
            x, tg = ds[id2idx[iid]]
            d = pp(model(x.unsqueeze(0).cuda()), tg['orig_size'].unsqueeze(0).cuda())[0]
            b, l, s = shown(d, args.known_thr, args.unk_thr)
            ug = torch.tensor([[a['bbox'][0], a['bbox'][1], a['bbox'][0] + a['bbox'][2], a['bbox'][1] + a['bbox'][3]]
                               for a in anns[iid] if a['category_id'] >= K])
            ub = b[l == K]
            if not len(ub) or not len(b):
                continue
            hits = int((torchvision.ops.box_iou(ug, ub).max(1).values >= 0.5).sum())
            iou = torchvision.ops.box_iou(b, b).fill_diagonal_(0)
            # at least --clean-frac of the shown boxes match a GT object of their label (and >= --min-boxes are shown) (known class, or any unknown-class object)
            allg = torch.tensor([[a['bbox'][0], a['bbox'][1], a['bbox'][0] + a['bbox'][2], a['bbox'][1] + a['bbox'][3]]
                                 for a in anns[iid]])
            glab = torch.tensor([min(a['category_id'], K) for a in anns[iid]])
            ok = (torchvision.ops.box_iou(b, allg) >= 0.5) & (l[:, None] == glab[None])
            clean = float(ok.any(1).float().mean()) >= args.clean_frac and len(b) >= args.min_boxes
            why['clean'] += clean
            why['hits'] += hits >= args.min_unk_hits
            why['nodup'] += bool(iou.max() < args.dup_iou)
            why['few'] += len(b) <= args.max_boxes
            if hits >= args.min_unk_hits and iou.max() < args.dup_iou and len(b) <= args.max_boxes and (clean or not args.clean):
                good.append((iid, b.tolist(), l.tolist(), s.tolist()))
    print(f'{len(good)} / {len(cand)} candidate images pass the filter; per condition: {dict(why)}')
    if args.list:
        print('good ids', ','.join(str(g[0]) for g in good))
    if args.ids:
        want = [int(i) for i in args.ids.split(',')]
        pick = [g for i in want for g in good if g[0] == i]
    else:
        pick = sorted(random.Random(args.seed).sample(good, min(args.num, len(good))), key=lambda g: g[0])

    fm.fontManager.addfont('/usr/share/fonts/truetype/lato/Lato-Bold.ttf')
    plt.rcParams['font.family'] = 'Lato'
    imgs = [Image.open(info[g[0]]['file_name']).convert('RGB') for g in pick]
    H = 400   # common display height; widths follow the aspect ratios
    ws = [im.width * H / im.height for im in imgs]
    fig = plt.figure(figsize=(sum(ws) / 100 + 0.1 * (len(imgs) - 1), H / 100), facecolor='white')
    x0 = 0.
    tot = sum(ws) + 10 * (len(imgs) - 1)
    for im, w, (iid, boxes, labels, scores) in zip(imgs, ws, pick):
        ax = fig.add_axes([x0 / tot, 0, w / tot, 1])
        x0 += w + 10
        ax.imshow(im)
        ax.set_xlim(0, im.width); ax.set_ylim(im.height, 0)
        ax.axis('off')
        lw = 2.2 * im.height / H
        for (x1, y1, x2, y2), l, s in sorted(zip(boxes, labels, scores), key=lambda t: t[2]):
            unk = l == K
            c = UNK_COLOR if unk else KNOWN_COLOR
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(im.width, x2), min(im.height, y2)
            ax.add_patch(mpatches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=c, linewidth=lw))
            ax.text(x1, y1, UNKNOWN_NAME if unk else ALL_CLASSES[l], fontsize=7.5, color='white',
                    fontweight='bold', va='top' if y1 < 0.06 * im.height else 'bottom', clip_on=True,
                    bbox=dict(facecolor=c, alpha=0.9, pad=1.2, edgecolor='none'))
        print('image', iid, os.path.basename(info[iid]['file_name']))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=200, facecolor='white')
    print('saved', args.out)


if __name__ == '__main__':
    main()
