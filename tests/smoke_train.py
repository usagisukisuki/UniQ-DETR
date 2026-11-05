"""Smoke test: a few train iterations + eval forward for a config (small batch, small GPU)."""
import os, sys, math, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import owod.register  # noqa
from src.core import YAMLConfig
from owod.eval.owod_eval import OWODEvaluator

ap = argparse.ArgumentParser(); ap.add_argument('config'); ap.add_argument('--iters', type=int, default=6)
ap.add_argument('--bs', type=int, default=2); ap.add_argument('--interval', type=int, default=2)
a = ap.parse_args()
cfg = YAMLConfig(a.config, **{'train_dataloader': {'total_batch_size': a.bs, 'num_workers': 2},
                             'val_dataloader': {'total_batch_size': a.bs, 'num_workers': 2},
                             'OWRTDETRTransformerv2': {'obj_interval': a.interval}})
dev = torch.device('cuda')
model, crit, post = cfg.model.to(dev), cfg.criterion.to(dev), cfg.postprocessor.to(dev)
opt = cfg.optimizer
loader = cfg.train_dataloader
model.train()
scaler = torch.amp.GradScaler()
t0 = time.time()
for i, (x, tg) in enumerate(loader):
    if i == a.iters: break
    x = x.to(dev); tg = [{k: v.to(dev) for k, v in t.items()} for t in tg]
    with torch.autocast('cuda'):
        out = model(x, targets=tg)
    with torch.autocast('cuda', enabled=False):
        ld = crit(out, tg)
    loss = sum(ld.values())
    assert math.isfinite(loss.item()), ld
    scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); opt.zero_grad()
    obj = {k: round(v.item(), 4) for k, v in ld.items() if 'obj' in k and 'aux' not in k}
    print(f'iter {i} loss {loss.item():.3f} n_terms {len(ld)} {obj}')
dec = model.decoder
if getattr(dec, 'use_objectness', False):
    for n, h in list(enumerate(dec.dec_obj_head))[-1:] + ([('enc', dec.enc_obj_head)] if hasattr(dec, 'enc_obj_head') else []):
        print(f'gauss {n}: init={bool(h.initialized)} |mu|={h.mean.norm():.3f} tr(cov)={h.cov.trace():.2f} fallbacks={h.num_fallbacks}')
print(f'train {a.iters} iters {time.time()-t0:.1f}s, max mem {torch.cuda.max_memory_allocated()/2**30:.2f} GB')

# eval
model.eval()
ev = OWODEvaluator('data/mowodb/test.json', num_known=20)
with torch.no_grad():
    for i, (x, tg) in enumerate(cfg.val_dataloader):
        if i == 3: break
        out = model(x.to(dev))
        res = post(out, torch.stack([t['orig_size'] for t in tg]).to(dev))
        ev.update({t['image_id'].item(): r for t, r in zip(tg, res)})
        if 'pred_objectness' in out:
            print('objectness range', out['pred_objectness'].min().item(), out['pred_objectness'].max().item())
print('labels seen', sorted({int(l) for r in res for l in r['labels']})[:25], 'n per img', len(res[0]['labels']))
print('SMOKE OK')
