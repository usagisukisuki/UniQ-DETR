"""Where is each unknown GT lost? Funnel over the whole test set (one image's trace: README visualisation, bananas).

Stages, IoU >= 0.5 with the GT:
  enc-all  some encoder proposal (one per token)
  enc-sel  one of the num_queries proposals selected by the encoder score (the decoder's initial boxes)
  dec      some final decoder box
  out      an 'unknown' detection among the post-processed top-100; 'found' also requires the VOC matching of the
           evaluator (a detection claims only its max-IoU GT), so found / all ~ U-Recall
Each unknown GT is assigned to the first stage it fails ('found' when it passes all). For GT lost at the top-100 cut,
the rank of its best unknown entry in the full (query x class) ranking and whether a known-class top-100 entry covers
it. For GT lost at query selection, the encoder-score rank of its best overlapping proposal (selection keeps the top
num_queries by max-class encoder score). Every stage is also split by GT box size (COCO bins on the box area in
original pixels: small < 32^2, medium < 96^2, large). Also the mean composition of the top-100 outputs (label x what it hits).

  uv run python tools/diag_unk_funnel.py -c configs/uniq_detr.yml -r outputs/g4_8/best.pth --pp unk_weight=12
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
import torchvision  # noqa: E402
from tqdm import tqdm  # noqa: E402

import owod.register  # noqa: F401,E402
from src.core import YAMLConfig  # noqa: E402


def iou(a, b):
    return torchvision.ops.box_iou(a, b) if len(a) and len(b) else torch.zeros(len(a), len(b), device=a.device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-c', '--config', required=True)
    ap.add_argument('-r', '--resume', required=True)
    ap.add_argument('--pp', nargs='*', default=[], help='OWPostProcessor overrides, e.g. unk_weight=12')
    ap.add_argument('--max-images', type=int, default=None)
    ap.add_argument('--dec', nargs='*', default=[], help='OWRTDETRTransformerv2 overrides, e.g. unk_select=100')
    args = ap.parse_args()
    pp_over = {k: None if v == 'None' else float(v) for k, v in (a.split('=') for a in args.pp)}
    dec_over = {k: int(v) if v.isdigit() else v for k, v in (a.split('=') for a in args.dec)}
    cfg = YAMLConfig(args.config, OWPostProcessor=pp_over, val_dataloader={'total_batch_size': 16},
                     **({'OWRTDETRTransformerv2': dec_over} if dec_over else {}))
    model = cfg.model
    model.load_state_dict(torch.load(args.resume, map_location='cpu')['ema']['module'])
    model = model.cuda().eval()
    pp, dec = cfg.postprocessor, model.decoder
    K = cfg.yaml_cfg['owod_eval']['num_known']
    top_n = pp.num_top_queries
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
        if not a.get('difficult', 0):
            anns[a['image_id']].append(a)
    # owod_all_task_test.txt lists 1,127 images twice; the evaluator (as PROB) counts their GT twice but can match
    # only the first listing. The funnel uses each image once and also reports the evaluator-equivalent ratio.
    name = {i['id']: i.get('owod_name', i['id']) for i in gt['images']}
    n_unk_listed = sum(a['category_id'] >= K for v in anns.values() for a in v)
    seen = set()
    stage = Counter()
    lost_rank, lost_known = [], 0
    sel_rank = []
    by_size = defaultdict(Counter)
    size_of = lambda a: 'small' if a < 32 ** 2 else ('medium' if a < 96 ** 2 else 'large')
    comp = Counter()
    n_img = 0
    with torch.no_grad():
        for x, tg in tqdm(cfg.val_dataloader, desc='funnel'):
            out = model(x.cuda())
            sizes = torch.stack([t['orig_size'] for t in tg]).cuda()
            pp.num_top_queries = out['pred_logits'].shape[1] * (K + 1)   # full ranking; first top_n = real output
            res = pp(out, sizes)
            pp.num_top_queries = top_n
            props_all = torch.sigmoid(tap['coord'] + dec.anchors.to(tap['coord'].device))
            valid = dec.valid_mask[0, :, 0].to(props_all.device)
            for j, t in enumerate(tg):
                iid = t['image_id'].item()
                if name[iid] in seen:
                    continue
                seen.add(name[iid])
                w, h = t['orig_size'].tolist()
                sc = torch.tensor([w, h, w, h], device=props_all.device, dtype=torch.float32)
                xyxy = lambda b: torchvision.ops.box_convert(b, 'cxcywh', 'xyxy') * sc
                pa = xyxy(props_all[j])
                pa = torch.where(valid[:, None], pa, torch.zeros_like(pa))
                ps = pa[tap['ind'][j]]
                enc_rank = torch.empty(len(pa), dtype=torch.long, device=pa.device)
                enc_rank[tap['logits'][j].max(-1).values.argsort(descending=True)] = torch.arange(len(pa), device=pa.device)
                pd = xyxy(out['pred_boxes'][j].float())
                r = res[j]
                ql, lab, box = r['query_index'], r['labels'], r['boxes'].float()
                A = anns[iid]
                G = torch.tensor([[a['bbox'][0], a['bbox'][1], a['bbox'][0] + a['bbox'][2], a['bbox'][1] + a['bbox'][3]]
                                  for a in A], dtype=torch.float32, device=pd.device).reshape(-1, 4)
                gunk = torch.tensor([a['category_id'] >= K for a in A], dtype=torch.bool, device=pd.device)
                # composition of the real top-n output
                io = iou(box[:top_n], G)
                bv, bi = io.max(1) if len(G) else (torch.zeros(min(top_n, len(box)), device=pd.device), None)
                for i in range(min(top_n, len(box))):
                    l = 'unknown' if lab[i] == K else 'known'
                    hit = 'background' if bv[i] < 0.5 else ('unknown GT' if gunk[bi[i]] else 'known GT')
                    comp[f'{l} label on {hit}'] += 1
                n_img += 1
                U = G[gunk]
                usz = [size_of(float((b[2] - b[0]) * (b[3] - b[1]))) for b in U]
                if not len(U):
                    continue
                ia, isel, idec = iou(U, pa), iou(U, ps), iou(U, pd)
                iout = iou(U, box[:top_n])
                unk_out = (lab[:top_n] == K)
                # VOC matching (as the evaluator): an unknown detection can only claim its max-IoU GT
                iu = iout[:, unk_out]
                claimed = set(iu.argmax(0)[iu.max(0).values > 0.5].tolist()) if iu.numel() else set()
                for g in range(len(U)):
                    before = dict(stage)
                    if g in claimed:
                        stage['found'] += 1
                    elif (iout[g] >= 0.5)[unk_out].any():
                        stage['covered, but every covering box claims another GT'] += 1
                    elif (idec[g] >= 0.5).any():
                        stage['lost at top-100 cut (dec box exists)'] += 1
                        qok = torch.where(idec[g] >= 0.5)[0]
                        ru = (torch.isin(ql, qok) & (lab == K)).nonzero()
                        lost_rank.append(ru[0, 0].item() if len(ru) else 10 ** 6)
                        lost_known += bool((iout[g] >= 0.5)[~unk_out].any())
                    elif (isel[g] >= 0.5).any():
                        stage['lost in decoder (selected proposal, box drifted)'] += 1
                    elif (ia[g] >= 0.5).any():
                        stage['lost at query selection (proposal not selected)'] += 1
                        sel_rank.append(enc_rank[ia[g] >= 0.5].min().item())
                    else:
                        stage['no encoder proposal'] += 1
                    by_size[usz[g]][next(k for k in stage if stage[k] != before.get(k, 0))] += 1
            if args.max_images and n_img >= args.max_images:
                break
    tot = max(sum(stage.values()), 1)
    print(f'{os.path.basename(args.config)} {pp_over} {dec_over} | unique images {n_img} | unknown GT {tot} '
          f'(listed {n_unk_listed}; found / listed = {100 * stage["found"] / n_unk_listed:.2f}%, cf. U-Recall)')
    ORDER = ('found', 'covered, but every covering box claims another GT', 'lost at top-100 cut (dec box exists)', 'lost in decoder (selected proposal, box drifted)',
              'lost at query selection (proposal not selected)', 'no encoder proposal')
    for k in ORDER:
        print(f'  {k:50s} {stage[k]:6d}  {100 * stage[k] / tot:5.1f}%')
    if lost_rank:
        lr = torch.tensor(lost_rank, dtype=torch.float32)
        bins = [(top_n, 2 * top_n), (2 * top_n, 5 * top_n), (5 * top_n, 10 ** 6), (10 ** 6, 10 ** 7)]
        print('  lost at top-100 cut: best unknown rank ' + ', '.join(
            f'[{a},{b}) {100 * ((lr >= a) & (lr < b)).float().mean():.0f}%' for a, b in bins)
            + f' | covered by a known-label top-100 entry {100 * lost_known / len(lost_rank):.0f}%')
    if sel_rank:
        sr = torch.tensor(sel_rank, dtype=torch.float32)
        nq = dec.num_queries
        bins = [(nq, int(1.5 * nq)), (int(1.5 * nq), 2 * nq), (2 * nq, 4 * nq), (4 * nq, 8 * nq), (8 * nq, 10 ** 6)]
        print(f'  lost at query selection: best overlapping proposal encoder rank (selected = top {nq}) ' + ', '.join(
            f'[{a},{b}) {100 * ((sr >= a) & (sr < b)).float().mean():.0f}%' for a, b in bins)
            + f' | median {sr.median().item():.0f}')
    print('  by GT size (share of each stage within the size bin):')
    for z in ('small', 'medium', 'large'):
        n = max(sum(by_size[z].values()), 1)
        print(f'    {z:6s} n={n:5d} ({100 * n / tot:4.1f}%): ' + ', '.join(
            f'{k.split(" (")[0]} {100 * by_size[z][k] / n:.1f}%' for k in ORDER))
    ctot = sum(comp.values())
    print(f'  top-{top_n} composition (per image): ' + ', '.join(
        f'{k} {comp[k] / n_img:.1f}' for k in sorted(comp, key=lambda k: -comp[k])) + f' (total {ctot / n_img:.1f})')


if __name__ == '__main__':
    main()
