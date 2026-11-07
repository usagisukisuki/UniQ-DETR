"""Where are small unknown objects lost? (analysis/G4_PLAN.md 6.6, small-object coverage)

COCO-source test images. For every unknown GT (and, for contrast, every known GT), by COCO size (small < 32^2 px,
medium < 96^2, large), coverage = some box has IoU >= 0.5 with it, at three stages:
  enc-all  all encoder proposals (one per encoder token: 80x80 + 40x40 + 20x20 at 640)
  enc-sel  the num_queries proposals selected by the encoder score (max over the class logits) = the decoder's initial boxes
  dec      the final decoder boxes (also at IoU >= 0.3)
For unknown GT covered by enc-all but not enc-sel: the selection rank of its best proposal, its stride level, and the
max known / unknown encoder logit compared with the selected proposals.

  uv run python tools/diag_small_cov.py -c configs/uniq_detr.yml -r outputs/g4_6/best.pth [--nq 900] [--size 640]
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

SIZES = ('small', 'medium', 'large')


def iou(a, b):
    return torchvision.ops.box_iou(a, b) if len(a) and len(b) else torch.zeros(len(a), len(b), device=a.device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-c', '--config', required=True)
    ap.add_argument('-r', '--resume', required=True)
    ap.add_argument('--nq', type=int, default=None, help='decoder queries at inference (default: the config)')
    ap.add_argument('--size', type=int, default=640, help='inference input size (square)')
    ap.add_argument('--max-images', type=int, default=2008)
    args = ap.parse_args()
    over = {'val_dataloader': {'total_batch_size': 16}}
    if args.size != 640:
        over['eval_spatial_size'] = [args.size, args.size]
        over['val_dataloader']['dataset'] = {'transforms': {'type': 'Compose', 'ops': [
            {'type': 'Resize', 'size': [args.size, args.size]}, {'type': 'ConvertPILImage', 'dtype': 'float32', 'scale': True}]}}
    if args.nq:
        over['OWRTDETRTransformerv2'] = {'num_queries': args.nq}
    cfg = YAMLConfig(args.config, **over)
    model = cfg.model
    sd = torch.load(args.resume, map_location='cpu')['ema']['module']
    sd = {k: v for k, v in sd.items() if not k.endswith(('anchors', 'valid_mask', 'pos_embed2'))}
    print('missing keys:', [k for k in model.load_state_dict(sd, strict=False).missing_keys])
    model = model.cuda().eval()
    dec = model.decoder
    K = cfg.yaml_cfg['owod_eval']['num_known']
    tap = {}
    dec.enc_score_head.register_forward_hook(lambda m, i, o: tap.__setitem__('logits', o.float()))
    dec.enc_bbox_head.register_forward_hook(lambda m, i, o: tap.__setitem__('coord', o.float()))
    topk_index = dec._topk_index

    def capture(logits):
        ind = topk_index(logits)
        tap['ind'] = ind
        return ind
    dec._topk_index = capture
    gt = json.load(open(cfg.yaml_cfg['owod_eval']['gt_json']))
    anns = defaultdict(list)
    for a in gt['annotations']:
        anns[a['image_id']].append(a)
    src = {i['id']: i['source'] for i in gt['images']}
    cov = {kind: {s: [0, 0, 0, 0, 0] for s in SIZES} for kind in ('unknown', 'known')}   # n, all, sel, dec, dec@0.3
    lost = []                                           # unknown GT in enc-all but not enc-sel: rank, level, knownmax, unk
    sel_stats = []
    n = 0
    levels = None
    with torch.no_grad():
        for x, tg in cfg.val_dataloader:
            keep = [j for j, t in enumerate(tg) if src[t['image_id'].item()] == 'coco']
            if not keep:
                continue
            out = model(x.cuda())
            if levels is None:
                hw = [(args.size // s) ** 2 for s in (8, 16, 32)]
                levels = torch.cat([torch.full((h,), i) for i, h in enumerate(hw)]).cuda()
            anchors = dec.anchors.to(tap['coord'].device)
            props = torch.sigmoid(tap['coord'] + anchors)                                  # [B, L, 4]
            logits = tap['logits']
            sel_score = logits.max(-1).values
            order = sel_score.argsort(-1, descending=True).argsort(-1)                     # rank per proposal
            for j in keep:
                iid = tg[j]['image_id'].item()
                w, h = tg[j]['orig_size'].tolist()
                sc = torch.tensor([w, h, w, h], device=props.device, dtype=torch.float32)
                xyxy = lambda b: torchvision.ops.box_convert(b, 'cxcywh', 'xyxy') * sc
                valid = dec.valid_mask[0, :, 0].to(props.device)
                pa = xyxy(props[j])
                pa = torch.where(valid[:, None], pa, torch.zeros_like(pa))
                ps = pa[tap['ind'][j]]
                pd = xyxy(out['pred_boxes'][j].float())
                sel_stats.append(torch.stack([logits[j, tap['ind'][j], :K].max(-1).values,
                                              logits[j, tap['ind'][j], K:].max(-1).values if logits.shape[-1] > K
                                              else torch.full_like(tap['ind'][j], float('nan'), dtype=torch.float32)], -1).cpu())
                for kind, A in (('unknown', [a for a in anns[iid] if a['category_id'] >= K]),
                                ('known', [a for a in anns[iid] if a['category_id'] < K])):
                    if not A:
                        continue
                    g = torch.tensor([[a['bbox'][0], a['bbox'][1], a['bbox'][0] + a['bbox'][2], a['bbox'][1] + a['bbox'][3]]
                                      for a in A], dtype=torch.float32, device=props.device)
                    ia, isel, idec = iou(g, pa), iou(g, ps), iou(g, pd)
                    for gi, a in enumerate(A):
                        ar = a['bbox'][2] * a['bbox'][3]
                        s = 'small' if ar < 32 ** 2 else 'medium' if ar < 96 ** 2 else 'large'
                        c = cov[kind][s]
                        call, csel = bool(ia[gi].max() >= 0.5), bool(isel[gi].max() >= 0.5)
                        c[0] += 1; c[1] += call; c[2] += csel
                        c[3] += bool(idec[gi].max() >= 0.5); c[4] += bool(idec[gi].max() >= 0.3)
                        if kind == 'unknown' and call and not csel:
                            cand = torch.where(ia[gi] >= 0.5)[0]
                            b = cand[order[j, cand].argmin()]
                            lost.append([s == 'small', order[j, b].item(), levels[b].item(),
                                         logits[j, b, :K].max().item(),
                                         logits[j, b, K:].max().item() if logits.shape[-1] > K else float('nan')])
            n += len(keep)
            if n >= args.max_images:
                break
    nq = tap['ind'].shape[1]
    print(f'{os.path.basename(args.config)} size {args.size} queries {nq} proposals {props.shape[1]} | images {n}')
    for kind in ('unknown', 'known'):
        for s in SIZES:
            m, ca, cs, cd, cd3 = cov[kind][s]
            if m:
                print(f'  {kind:7s} {s:6s} GT {m:5d} | enc-all {100 * ca / m:5.1f}% | enc-sel {100 * cs / m:5.1f}% | '
                      f'dec {100 * cd / m:5.1f}% (IoU>=0.3 {100 * cd3 / m:5.1f}%)')
    st = torch.cat(sel_stats)
    print(f'  selected proposals: median max known logit {st[:, 0].median():.2f}, unknown logit {st[:, 1].nanmedian():.2f}')
    if lost:
        L = torch.tensor(lost)
        for name, msk in (('small', L[:, 0] == 1), ('medium+large', L[:, 0] == 0)):
            r = L[msk]
            if not len(r):
                continue
            lv = [100 * (r[:, 2] == i).float().mean().item() for i in range(3)]
            print(f'  unknown {name:12s} in enc-all but not selected: {len(r)} | rank median {r[:, 1].median():.0f} '
                  f'(<2x queries {100 * (r[:, 1] < 2 * nq).float().mean():.0f}%) | level s8/s16/s32 '
                  f'{lv[0]:.0f}/{lv[1]:.0f}/{lv[2]:.0f}% | max known logit {r[:, 3].median():.2f} unknown logit {r[:, 4].nanmedian():.2f}')


if __name__ == '__main__':
    main()
