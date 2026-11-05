"""Diagnose why C3 / C4 do not raise U-Recall (docs/C3C4_ANALYSIS.md).

On test images from COCO (the only ones with unknown GT), for one checkpoint:
  enc   coverage of known / unknown GT by the encoder-selected query boxes (300 by default; C4 acts here)
  dec   coverage by the final decoder boxes = upper bound of recall before top-100 ranking
  top   fraction of unknown GT whose best-covering query makes it into the top-100 as 'unknown'
        (the postprocessor of the config, incl. unk_weight)
  C3    among queries not overlapping any known GT (IoU<0.5), the k_bg with the highest
        objectness (what C3 masks): fraction that cover an unknown GT, vs. all such queries
  stats on the best-covering query of each unknown GT: o, sigmoid(u), max_c p(c)

  uv run python tools/diag_c3c4.py -c configs/e2_obj.yml -r outputs/e2_obj/best.pth
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


def iou(a, b):
    return torchvision.ops.box_iou(a, b) if len(a) and len(b) else torch.zeros(len(a), len(b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-c', '--config', required=True)
    ap.add_argument('-r', '--resume', required=True)
    ap.add_argument('--max-images', type=int, default=2000)
    ap.add_argument('--k-bg', type=int, default=5)
    ap.add_argument('--bs', type=int, default=16)
    ap.add_argument('--num-queries', type=int, default=None, help='override decoder queries at inference (V2)')
    args = ap.parse_args()

    upd = {'val_dataloader': {'total_batch_size': args.bs, 'shuffle': False}}
    if args.num_queries:
        upd['OWRTDETRTransformerv2'] = {'num_queries': args.num_queries}
    cfg = YAMLConfig(args.config, **upd)
    state = torch.load(args.resume, map_location='cpu')
    model = cfg.model
    model.load_state_dict(state['ema']['module'])
    model = model.cuda().eval()
    pp = cfg.postprocessor
    K = cfg.yaml_cfg['num_classes']

    # capture the encoder-selected reference boxes (input of the decoder)
    dec = model.decoder
    cap = {}
    orig = dec._get_decoder_input

    def hooked(*a, **kw):
        out = orig(*a, **kw)
        cap['enc_boxes'] = out[1].sigmoid()        # [B, 300, 4] cxcywh normalised
        return out
    dec._get_decoder_input = hooked

    gt = json.load(open(cfg.yaml_cfg['owod_eval']['gt_json']))
    anns = defaultdict(list)
    for a in gt['annotations']:
        anns[a['image_id']].append(a)
    src = {i['id']: i['source'] for i in gt['images']}

    acc = defaultdict(float)
    best_stats = defaultdict(list)
    n_img = 0
    with torch.no_grad():
        for x, tg in cfg.val_dataloader:
            keep = [j for j, t in enumerate(tg) if src[t['image_id'].item()] == 'coco']
            if not keep:
                continue
            out = model(x.cuda())
            sizes = torch.stack([t['orig_size'] for t in tg]).cuda()
            dets = pp(out, sizes)
            for j in keep:
                iid = tg[j]['image_id'].item()
                w, h = tg[j]['orig_size'].tolist()
                scale = torch.tensor([w, h, w, h], dtype=torch.float32)
                to_xyxy = lambda b: torchvision.ops.box_convert(b.float().cpu(), 'cxcywh', 'xyxy') * scale
                gk = [a for a in anns[iid] if a['category_id'] < K and not a['difficult']]
                gu = [a for a in anns[iid] if a['category_id'] >= K]
                box = lambda L: torch.tensor([[a['bbox'][0], a['bbox'][1], a['bbox'][0] + a['bbox'][2],
                                               a['bbox'][1] + a['bbox'][3]] for a in L], dtype=torch.float32).reshape(-1, 4)
                gk, gu = box(gk), box(gu)
                enc_b = to_xyxy(cap['enc_boxes'][j])
                dec_b = to_xyxy(out['pred_boxes'][j])
                o = out['pred_objectness'][j].float().cpu()
                pu = out['pred_unk_logits'][j].float().sigmoid().cpu() if 'pred_unk_logits' in out else torch.zeros_like(o)
                pmax = out['pred_logits'][j].float().sigmoid().max(-1).values.cpu()

                for name, g in (('known', gk), ('unk', gu)):
                    acc[f'n_{name}'] += len(g)
                    if len(g):
                        acc[f'enc_cov_{name}'] += float((iou(g, enc_b).max(1).values >= 0.5).sum())
                        acc[f'dec_cov_{name}'] += float((iou(g, dec_b).max(1).values >= 0.5).sum())

                if len(gu):
                    m = iou(gu, dec_b)
                    best_v, best_q = m.max(1)
                    cov = best_v >= 0.5
                    # top-100 'unknown' detections of this image (as evaluated)
                    d = dets[j]
                    ub = d['boxes'][d['labels'] == K].float().cpu()
                    hit = (iou(gu, ub).max(1).values >= 0.5) if len(ub) else torch.zeros(len(gu), dtype=torch.bool)
                    acc['top_unk'] += float(hit.sum())
                    acc['cov_but_missed'] += float((cov & ~hit).sum())
                    kb = d['boxes'][d['labels'] < K].float().cpu()
                    acc['unk_as_known'] += float((iou(gu, kb).max(1).values >= 0.5).sum()) if len(kb) else 0.
                    for q in best_q[cov]:
                        best_stats['o'].append(float(o[q])); best_stats['su'].append(float(pu[q]))
                        best_stats['pmax'].append(float(pmax[q]))

                # C3 target set: queries not overlapping known GT; top-k_bg by objectness
                free = (iou(dec_b, gk).max(1).values < 0.5) if len(gk) else torch.ones(len(dec_b), dtype=torch.bool)
                fi = free.nonzero().squeeze(1)
                if len(fi):
                    on_unk = (iou(dec_b[fi], gu).max(1).values >= 0.5) if len(gu) else torch.zeros(len(fi), dtype=torch.bool)
                    top = o[fi].topk(min(args.k_bg, len(fi))).indices
                    acc['c3_sel'] += len(top); acc['c3_sel_unk'] += float(on_unk[top].sum())
                    acc['free'] += len(fi); acc['free_unk'] += float(on_unk.sum())
                # objectness of all queries vs queries on known / unknown objects
                acc['o_all'] += float(o.sum()); acc['n_q'] += len(o)
                n_img += 1
            if n_img >= args.max_images:
                break

    f = lambda a, b: 100 * acc[a] / max(acc[b], 1)
    mean = lambda k: sum(best_stats[k]) / max(len(best_stats[k]), 1)
    print(f'images {n_img} | known GT {acc["n_known"]:.0f} | unknown GT {acc["n_unk"]:.0f}')
    print(f'enc-selected 300 cover: known {f("enc_cov_known", "n_known"):.1f}%  unknown {f("enc_cov_unk", "n_unk"):.1f}%')
    print(f'decoder 300 cover     : known {f("dec_cov_known", "n_known"):.1f}%  unknown {f("dec_cov_unk", "n_unk"):.1f}%')
    print(f'unknown GT hit by top-100 unknown dets {f("top_unk", "n_unk"):.1f}% | covered but missed {f("cov_but_missed", "n_unk"):.1f}% '
          f'| hit by a known-class det {f("unk_as_known", "n_unk"):.1f}%')
    print(f'best query on covered unknown GT: o {mean("o"):.3f} sigmoid(u) {mean("su"):.3f} max p {mean("pmax"):.3f} '
          f'| all queries o {acc["o_all"] / max(acc["n_q"], 1):.3f}')
    print(f'C3 set (top-{args.k_bg} objectness among non-known queries) on unknown GT {f("c3_sel_unk", "c3_sel"):.1f}% '
          f'vs all non-known queries {f("free_unk", "free"):.1f}%')


if __name__ == '__main__':
    main()
