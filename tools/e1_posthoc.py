"""E1 / E-score: post-hoc unknown scores on a trained checkpoint (no training).

1. (E1 only, --fit-gaussian) Fit the decoder last-layer Gaussian on the *training* set:
   eval-mode forward (no augmentation), Hungarian-match to known GT, pool matched query
   embeddings, set (mu, Sigma + lambda I). Uses known GT only.
2. One pass over the test set; every requested unknown-score function is evaluated
   from the same raw outputs (MSP / Energy / Mahalanobis / C1).

  uv run python tools/e1_posthoc.py -c configs/e0_closed.yml -r outputs/e0_closed/best.pth --fit-gaussian
  uv run python tools/e1_posthoc.py -c configs/e4_m0.yml    -r outputs/e4_m0/best.pth      # E-score
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


def load_model(cfg, ckpt):
    model = cfg.model
    state = torch.load(ckpt, map_location='cpu')
    sd = state['ema']['module'] if 'ema' in state else state['model']
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not unexpected, unexpected
    assert all('obj_head' in k for k in missing), missing  # only Gaussian buffers may be missing (E0)
    return model


@torch.no_grad()
def fit_gaussian(cfg, model, device, max_images=None):
    """Pool matched last-layer embeddings over the training set (no augmentation)."""
    import copy
    dl_cfg = copy.deepcopy(cfg.yaml_cfg['val_dataloader'])
    tr = cfg.yaml_cfg['train_dataloader']['dataset']
    dl_cfg['dataset'].update(ann_file=tr['ann_file'], split='train', num_known=tr['num_known'])
    cfg.yaml_cfg['gauss_fit_dataloader'] = dl_cfg
    loader = cfg.build_dataloader('gauss_fit_dataloader')
    matcher = cfg.criterion.matcher
    dec = model.decoder
    head = dec.dec_obj_head[dec.decoder.eval_idx]
    D = head.dim
    s = torch.zeros(D, dtype=torch.float64, device=device)
    o = torch.zeros(D, D, dtype=torch.float64, device=device)
    n, seen = 0, 0
    for x, tg in tqdm(loader, desc='fit gaussian'):
        x = x.to(device)
        out = model(x)
        # val transforms leave boxes as absolute xyxy in the resized image; matcher expects normalized cxcywh
        sz = torch.tensor([x.shape[-1], x.shape[-2]] * 2, dtype=torch.float32)
        tgs = []
        for t in tg:
            b = t['boxes'].float() / sz
            b = torch.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2, b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]], -1)
            tgs.append({'labels': t['labels'].to(device), 'boxes': b.to(device)})
        idx = matcher({'pred_logits': out['pred_logits'], 'pred_boxes': out['pred_boxes']}, tgs)['indices']
        emb = torch.cat([out['pred_embed'][i][src.to(device)] for i, (src, _) in enumerate(idx)]).double()
        s += emb.sum(0)
        o += emb.T @ emb
        n += emb.shape[0]
        seen += x.shape[0]
        if max_images and seen >= max_images:
            break
    mean = s / n
    cov = (o - n * torch.outer(mean, mean)) / (n - 1)
    if head.cov_type == 'diag':
        cov = torch.diag(cov.diagonal())
    cov += head.shrinkage * torch.eye(D, device=device, dtype=cov.dtype)
    head.mean.copy_(mean.float())
    head.cov.copy_(cov.float())
    head.initialized.fill_(True)
    print(f'fitted Gaussian on {n} matched queries from {seen} train images')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-c', '--config', required=True)
    ap.add_argument('-r', '--resume', required=True)
    ap.add_argument('--fit-gaussian', action='store_true')
    ap.add_argument('--scores', default='msp,energy,maha,c1')
    ap.add_argument('--known-score', default=None, help="'cls' for E1 (E0 has no trained objectness)")
    ap.add_argument('--out', default=None)
    ap.add_argument('--bs', type=int, default=32)
    ap.add_argument('--max-test-images', type=int, default=None)
    ap.add_argument('--max-fit-images', type=int, default=None)
    args = ap.parse_args()

    upd = {'OWRTDETRTransformerv2': {'use_objectness': True},
           'val_dataloader': {'total_batch_size': args.bs}}
    cfg = YAMLConfig(args.config, **upd)
    device = torch.device('cuda')
    model = load_model(cfg, args.resume).to(device).eval()
    model.decoder.return_embed = True
    if args.fit_gaussian:
        fit_gaussian(cfg, model, device, args.max_fit_images)

    known_score = args.known_score or ('cls' if args.fit_gaussian else 'obj_cls')
    names = args.scores.split(',')
    pps = {k: OWPostProcessor(num_classes=cfg.yaml_cfg['num_classes'], num_top_queries=100,
                              unk_score=k, known_score=known_score) for k in names}
    gt = json.load(open(cfg.yaml_cfg['owod_eval']['gt_json']))
    evs = {k: OWODEvaluator(gt, num_known=cfg.yaml_cfg['owod_eval']['num_known']) for k in names}
    seen = 0
    with torch.no_grad():
        for x, tg in tqdm(cfg.val_dataloader, desc='test'):
            out = model(x.to(device))
            sizes = torch.stack([t['orig_size'] for t in tg]).to(device)
            for k in names:
                res = pps[k](out, sizes)
                evs[k].update({t['image_id'].item(): r for t, r in zip(tg, res)})
            seen += x.shape[0]
            if args.max_test_images and seen >= args.max_test_images:
                break
    results = {}
    for k in names:
        print(f'--- unk_score={k} (known_score={known_score})')
        evs[k].accumulate()
        results[k] = evs[k].summarize()
    out = args.out or os.path.join(os.path.dirname(args.resume), 'posthoc_scores.json')
    with open(out, 'w') as f:
        json.dump(results, f, indent=1)
    print('saved', out)


if __name__ == '__main__':
    main()
