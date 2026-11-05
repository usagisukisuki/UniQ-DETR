"""V3: PROB (Deformable DETR) inference throughput with its official T1 weights, batch 1, 800x1067
(PROB's test resize of a 4:3 image: shorter side 800). Run with cwd = third_party/PROB and its venv:

  cd third_party/PROB && .venv/bin/python ../../tools/prob_check/bench_prob_fps.py
"""
import os
import sys
import time

import numpy as np
import torch

np.bool = bool
sys.path.insert(0, os.getcwd())
from main_open_world import get_args_parser  # noqa: E402
from models import build_model  # noqa: E402
from util.misc import nested_tensor_from_tensor_list  # noqa: E402

args = get_args_parser().parse_args([
    '--dataset', 'TOWOD', '--PREV_INTRODUCED_CLS', '0', '--CUR_INTRODUCED_CLS', '20', '--model_type', 'prob',
    '--obj_loss_coef', '8e-4', '--obj_temp', '1.3', '--wandb_project', ''])
model, _, post, _ = build_model(args, mode='prob')
ck = torch.load('../prob_download/MOWODB/t1.pth', map_location='cpu')
model.load_state_dict(ck['model'], strict=False)
model = model.cuda().eval()
torch.manual_seed(0)
x = nested_tensor_from_tensor_list([torch.rand(3, 800, 1067, device='cuda')])
sizes = torch.tensor([[480, 640]], device='cuda')


@torch.no_grad()
def bench(amp, warmup=50, iters=300):
    for _ in range(warmup):
        with torch.autocast('cuda', enabled=amp):
            post['bbox'](model(x), sizes)
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        with torch.autocast('cuda', enabled=amp):
            post['bbox'](model(x), sizes)
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters


for amp in (False, True):
    try:
        dt = bench(amp)
        print(f'PROB amp={amp}: {dt * 1000:.2f} ms/img  {1 / dt:.1f} FPS', flush=True)
    except Exception as e:  # MSDeformAttn CUDA op may not support fp16
        print(f'PROB amp={amp}: failed ({type(e).__name__}: {e})')
