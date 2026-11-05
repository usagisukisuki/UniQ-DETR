"""Train with tools/train.py but save checkpoints at given *iterations* (the solver only saves per epoch).

Hooks ModelEMA.update (called once per training iteration) and saves {'model', 'ema', 'iter'} at each
requested step to <output_dir>/iter{step:05d}.pth, then exits after the last one. Used for V1 at
iteration granularity (docs/NEXT_STEPS.md).

  uv run python tools/train_iter_ckpt.py --steps 50 100 500 1000 --out outputs/e2_iter -- -c configs/e2_obj.yml -u epoches=2 output_dir=./outputs/e2_iter
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

import train  # noqa: E402  (tools/train.py: sets up paths and registers owod)
from src.optim import ModelEMA  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--steps', type=int, nargs='+', required=True)
ap.add_argument('--out', required=True)
args, rest = ap.parse_known_args()
rest = [a for a in rest if a != '--']
steps, last = set(args.steps), max(args.steps)
os.makedirs(args.out, exist_ok=True)

_update = ModelEMA.update


def update(self, model):
    _update(self, model)
    it = int(self.updates)
    if it in steps:
        m = model.module if hasattr(model, 'module') else model
        torch.save({'model': m.state_dict(), 'ema': self.state_dict(), 'iter': it},
                   os.path.join(args.out, f'iter{it:05d}.pth'))
        print(f'[iter-ckpt] saved iteration {it}', flush=True)
        if it == last:
            os._exit(0)


ModelEMA.update = update
sys.argv = [sys.argv[0]] + rest
train.main(train.build_parser().parse_args())
