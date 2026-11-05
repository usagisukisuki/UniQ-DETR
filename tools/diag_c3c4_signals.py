"""Follow-up to diag_c3c4.py (docs/C3C4_ANALYSIS.md): can C3 / C4 be fixed without new supervision?

C4 (needs a checkpoint with query_select_method=objectness, i.e. E4): coverage of known / unknown GT
    by the encoder candidates when the default class-score top-300 is *augmented* with the n best
    tokens by objectness (n = 0 .. 300), vs objectness alone.
C3 (any checkpoint): among decoder queries that do not overlap a known GT (IoU<0.5), how well does
    each per-query signal separate queries on an unknown object (IoU>=0.5) from the rest
    (AUROC, and hit rate of the per-image top-k_bg).

  uv run python tools/diag_c3c4_signals.py -c configs/e4_m0.yml -r outputs/e4_m0/best.pth
"""
import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import torchvision  # noqa: E402

import owod.register  # noqa: F401,E402
from src.core import YAMLConfig  # noqa: E402

EXTRA = [0, 25, 50, 100, 200, 300]


def iou(a, b):
    return torchvision.ops.box_iou(a, b) if len(a) and len(b) else torch.zeros(len(a), len(b))


def auroc(score, label):
    """Mann-Whitney AUROC (ties counted half)."""
    pos, neg = score[label], score[~label]
    if len(pos) == 0 or len(neg) == 0:
        return float('nan')
    allv = torch.cat([pos, neg])
    r = torch.empty_like(allv)
    order = allv.argsort()
    r[order] = torch.arange(1, len(allv) + 1, dtype=allv.dtype)
    # average ranks for ties
    uniq, inv, cnt = torch.unique(allv, return_inverse=True, return_counts=True)
    sums = torch.zeros_like(uniq).scatter_add_(0, inv, r)
    r = (sums / cnt)[inv]
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-c', '--config', required=True)
    ap.add_argument('-r', '--resume', required=True)
    ap.add_argument('--max-images', type=int, default=2000)
    ap.add_argument('--k-bg', type=int, default=5)
    ap.add_argument('--bs', type=int, default=16)
    args = ap.parse_args()

    cfg = YAMLConfig(args.config, val_dataloader={'total_batch_size': args.bs, 'shuffle': False})
    state = torch.load(args.resume, map_location='cpu')
    model = cfg.model
    model.load_state_dict(state['ema']['module'])
    model = model.cuda().eval()
    K = cfg.yaml_cfg['num_classes']
    dec = model.decoder
    has_enc_obj = hasattr(dec, 'enc_obj_head')

    cap = {}
    orig = dec._get_decoder_input

    def hooked(memory, spatial_shapes, *a, **kw):
        anchors, valid = dec.anchors, dec.valid_mask
        om = dec.enc_output(valid.to(memory.dtype) * memory)
        cap['logit'] = dec.enc_score_head(om).max(-1).values.masked_fill(~valid.squeeze(-1), -1e9)
        cap['box'] = (dec.enc_bbox_head(om) + anchors).sigmoid()
        if has_enc_obj:
            cap['d2'] = dec.enc_obj_head.mahalanobis(om).masked_fill(~valid.squeeze(-1).expand(om.shape[:2]), 1e9)
        return orig(memory, spatial_shapes, *a, **kw)
    dec._get_decoder_input = hooked

    gt = json.load(open(cfg.yaml_cfg['owod_eval']['gt_json']))
    anns = defaultdict(list)
    for a in gt['annotations']:
        anns[a['image_id']].append(a)
    src = {i['id']: i['source'] for i in gt['images']}

    cov = defaultdict(float)
    sig = defaultdict(list)
    lab = []
    topk_hit = defaultdict(float)
    n_top, n_img = 0, 0
    with torch.no_grad():
        for x, tg in cfg.val_dataloader:
            keep = [j for j, t in enumerate(tg) if src[t['image_id'].item()] == 'coco']
            if not keep:
                continue
            out = model(x.cuda())
            for j in keep:
                iid = tg[j]['image_id'].item()
                w, h = tg[j]['orig_size'].tolist()
                scale = torch.tensor([w, h, w, h], dtype=torch.float32)
                xyxy = lambda b: torchvision.ops.box_convert(b.float().cpu(), 'cxcywh', 'xyxy') * scale
                box = lambda L: torch.tensor([[a['bbox'][0], a['bbox'][1], a['bbox'][0] + a['bbox'][2],
                                               a['bbox'][1] + a['bbox'][3]] for a in L], dtype=torch.float32).reshape(-1, 4)
                gk = box([a for a in anns[iid] if a['category_id'] < K and not a['difficult']])
                gu = box([a for a in anns[iid] if a['category_id'] >= K])
                cov['n_known'] += len(gk); cov['n_unk'] += len(gu)

                # ---------------- C4: augment class-score selection with objectness-selected tokens
                if has_enc_obj:
                    eb = xyxy(cap['box'][j])
                    by_cls = cap['logit'][j].argsort(descending=True).cpu()
                    by_obj = cap['d2'][j].argsort().cpu()
                    base = by_cls[:300]
                    in_base = torch.zeros(len(eb), dtype=torch.bool); in_base[base] = True
                    extra_pool = by_obj[~in_base[by_obj]]
                    for n in EXTRA:
                        sel = torch.cat([base, extra_pool[:n]])
                        for name, g in (('known', gk), ('unk', gu)):
                            if len(g):
                                cov[f'cls+{n}_{name}'] += float((iou(g, eb[sel]).max(1).values >= 0.5).sum())
                    for name, g in (('known', gk), ('unk', gu)):
                        if len(g):
                            cov[f'obj300_{name}'] += float((iou(g, eb[by_obj[:300]]).max(1).values >= 0.5).sum())
                            cov[f'cls600_{name}'] += float((iou(g, eb[by_cls[:600]]).max(1).values >= 0.5).sum())

                # ---------------- C3: signals on non-known decoder queries
                db = xyxy(out['pred_boxes'][j])
                free = (iou(db, gk).max(1).values < 0.5) if len(gk) else torch.ones(len(db), dtype=torch.bool)
                if free.sum() == 0:
                    continue
                y = (iou(db[free], gu).max(1).values >= 0.5) if len(gu) else torch.zeros(int(free.sum()), dtype=torch.bool)
                o = out['pred_objectness'][j].float().cpu()[free]
                su = out['pred_unk_logits'][j].float().sigmoid().cpu()[free]
                pmax = out['pred_logits'][j].float().sigmoid().max(-1).values.cpu()[free]
                area = ((db[:, 2] - db[:, 0]) * (db[:, 3] - db[:, 1]) / (w * h)).clamp(min=1e-6)[free]
                s = {'o': o, 'sigmoid(u)': su, '1-sigmoid(u)': 1 - su, 'max p': pmax, 'o*sigmoid(u)': o * su,
                     'o*max p': o * pmax, 'log area': area.log()}
                for k, v in s.items():
                    sig[k].append(v)
                    top = v.topk(min(args.k_bg, len(v))).indices
                    topk_hit[k] += float(y[top].sum())
                topk_hit['random'] += float(y.float().mean() * min(args.k_bg, len(y)))
                n_top += min(args.k_bg, len(y))
                lab.append(y)
                n_img += 1
            if n_img >= args.max_images:
                break

    print(f'images {n_img} | known GT {cov["n_known"]:.0f} | unknown GT {cov["n_unk"]:.0f}')
    if has_enc_obj:
        print('C4: encoder candidate coverage (IoU>=0.5)')
        for n in EXTRA:
            print(f'  class-score 300 + objectness {n:3d}: known {100 * cov[f"cls+{n}_known"] / cov["n_known"]:.1f}%  '
                  f'unknown {100 * cov[f"cls+{n}_unk"] / cov["n_unk"]:.1f}%')
        print(f'  class-score 600 (reference)      : known {100 * cov["cls600_known"] / cov["n_known"]:.1f}%  '
              f'unknown {100 * cov["cls600_unk"] / cov["n_unk"]:.1f}%')
        print(f'  objectness 300 only              : known {100 * cov["obj300_known"] / cov["n_known"]:.1f}%  '
              f'unknown {100 * cov["obj300_unk"] / cov["n_unk"]:.1f}%')
    y = torch.cat(lab)
    print(f'C3: non-known decoder queries {len(y)}, on an unknown object {100 * y.float().mean():.1f}%')
    for k, v in sig.items():
        v = torch.cat(v)
        print(f'  {k:14s} AUROC {auroc(v.double(), y):.3f} | top-{args.k_bg}/image on unknown {100 * topk_hit[k] / n_top:.1f}%')
    print(f'  {"random":14s}               | top-{args.k_bg}/image on unknown {100 * topk_hit["random"] / n_top:.1f}%')


if __name__ == '__main__':
    main()
