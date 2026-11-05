"""Visualize E1 post-hoc unknown scores on sample test images.

One figure per image: GT | msp | energy | maha | c1. Unknown-class GT and unknown
predictions are labelled "unknown". The Gaussian fitted on the training set is cached
next to the checkpoint so later runs skip the fit.

  uv run python tools/vis_e1.py -c configs/e0_closed.yml -r outputs/e0_closed/best.pth
"""
import argparse
import importlib.util
import os
import random
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
from owod.zoo.ow_postprocessor import OWPostProcessor  # noqa: E402

_spec = importlib.util.spec_from_file_location('e1_posthoc', os.path.join(os.path.dirname(__file__), 'e1_posthoc.py'))
e1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(e1)

KNOWN_COLOR = '#1f77b4'
UNK_COLOR = '#d62728'


def load_or_fit_gaussian(cfg, model, device, cache):
    head = model.decoder.dec_obj_head[model.decoder.decoder.eval_idx]
    if os.path.exists(cache):
        st = torch.load(cache, map_location=device)
        head.mean.copy_(st['mean'])
        head.cov.copy_(st['cov'])
        head.initialized.fill_(True)
        print('loaded Gaussian from', cache)
    else:
        e1.fit_gaussian(cfg, model, device)
        torch.save({'mean': head.mean.cpu(), 'cov': head.cov.cpu()}, cache)
        print('saved Gaussian to', cache)


def select_preds(res, num_known, known_thr, unk_thr, unk_topk, nms_iou):
    """Known: score >= known_thr. Unknown: NMS, drop boxes overlapping a shown known box, top-k."""
    labels, boxes, scores = res['labels'].cpu(), res['boxes'].cpu(), res['scores'].cpu()
    km = (labels < num_known) & (scores >= known_thr)
    kb, kl, ks = boxes[km], labels[km], scores[km]
    um = (labels == num_known) & (scores >= unk_thr)
    ub, us = boxes[um], scores[um]
    keep = torchvision.ops.nms(ub, us, nms_iou)
    ub, us = ub[keep], us[keep]
    if len(kb) and len(ub):
        ov = torchvision.ops.box_iou(ub, kb).max(1).values < nms_iou
        ub, us = ub[ov], us[ov]
    ub, us = ub[:unk_topk], us[:unk_topk]
    out = [(b, ALL_CLASSES[l], s, KNOWN_COLOR) for b, l, s in zip(kb.tolist(), kl.tolist(), ks.tolist())]
    out += [(b, UNKNOWN_NAME, s, UNK_COLOR) for b, s in zip(ub.tolist(), us.tolist())]
    return out


def draw(ax, img, items, title):
    W, H = img.size
    ax.imshow(img)
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_title(title, fontsize=12)
    ax.axis('off')
    for (x1, y1, x2, y2), name, s, color in items:
        # background queries can regress boxes outside the image; clip so panels stay aligned
        x1, x2 = max(0, min(x1, W)), max(0, min(x2, W))
        y1, y2 = max(0, min(y1, H)), max(0, min(y2, H))
        if x2 - x1 < 1 or y2 - y1 < 1:
            continue
        ax.add_patch(mpatches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, linewidth=2))
        txt = name if s is None else f'{name} {s:.2f}'
        ax.text(x1, y1, txt, fontsize=7, color='white', va='top' if y1 < 0.05 * H else 'bottom', clip_on=True,
                bbox=dict(facecolor=color, alpha=0.8, pad=1, edgecolor='none'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-c', '--config', required=True)
    ap.add_argument('-r', '--resume', required=True)
    ap.add_argument('--scores', default='msp,energy,maha,c1')
    ap.add_argument('--num-images', type=int, default=12)
    ap.add_argument('--image-ids', default=None, help='comma-separated test image ids (overrides sampling)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--known-thr', type=float, default=0.5)
    ap.add_argument('--unk-thr', type=float, default=0.0)
    ap.add_argument('--unk-topk', type=int, default=5)
    ap.add_argument('--nms-iou', type=float, default=0.5)
    ap.add_argument('--out-dir', default=None)
    args = ap.parse_args()

    cfg = YAMLConfig(args.config, OWRTDETRTransformerv2={'use_objectness': True})
    device = torch.device('cuda')
    model = e1.load_model(cfg, args.resume).to(device).eval()
    model.decoder.return_embed = True
    ckpt_dir = os.path.dirname(args.resume)
    load_or_fit_gaussian(cfg, model, device, os.path.join(ckpt_dir, 'e1_gaussian.pt'))

    num_known = cfg.yaml_cfg['owod_eval']['num_known']
    names = args.scores.split(',')
    pps = {k: OWPostProcessor(num_classes=cfg.yaml_cfg['num_classes'], num_top_queries=100,
                              unk_score=k, known_score='cls') for k in names}

    ds = cfg.val_dataloader.dataset
    coco = ds.coco
    if args.image_ids:
        img_ids = [int(i) for i in args.image_ids.split(',')]
    else:
        cand = [i for i in ds.ids
                if any(a['category_id'] >= num_known for a in coco.imgToAnns[i])
                and any(a['category_id'] < num_known for a in coco.imgToAnns[i])]
        img_ids = sorted(random.Random(args.seed).sample(cand, min(args.num_images, len(cand))))
    id2idx = {i: k for k, i in enumerate(ds.ids)}

    out_dir = args.out_dir or os.path.join(ckpt_dir, 'vis_e1')
    os.makedirs(out_dir, exist_ok=True)
    for img_id in img_ids:
        x, tg = ds[id2idx[img_id]]
        with torch.no_grad():
            out = model(x.unsqueeze(0).to(device))
        sizes = tg['orig_size'].unsqueeze(0).to(device)
        info = coco.imgs[img_id]
        img = Image.open(info['file_name']).convert('RGB')

        gt = []
        for a in coco.imgToAnns[img_id]:
            x0, y0, w, h = a['bbox']
            c = a['category_id']
            unk = c >= num_known
            name = f'{UNKNOWN_NAME} ({ALL_CLASSES[c]})' if unk else ALL_CLASSES[c]
            gt.append(([x0, y0, x0 + w, y0 + h], name, None, UNK_COLOR if unk else KNOWN_COLOR))

        n = len(names) + 1
        fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.5 * img.height / img.width + 0.8))
        draw(axes[0], img, gt, 'GT')
        for ax, k in zip(axes[1:], names):
            items = select_preds(pps[k](out, sizes)[0], num_known, args.known_thr, args.unk_thr,
                                 args.unk_topk, args.nms_iou)
            draw(ax, img, items, k)
        fig.suptitle(f'{os.path.basename(info["file_name"])} (id={img_id})  '
                     f'known: score>={args.known_thr}  unknown: top-{args.unk_topk} after NMS', fontsize=10)
        fig.tight_layout()
        path = os.path.join(out_dir, f'{img_id:06d}.png')
        fig.savefig(path, dpi=130, bbox_inches='tight')
        plt.close(fig)
        print('saved', path)


if __name__ == '__main__':
    main()
