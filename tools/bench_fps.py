"""V3: inference throughput (model + postprocessor), batch 1, 640x640, on the current GPU.

  uv run python tools/bench_fps.py -c configs/e2_obj.yml -r outputs/e2_obj/best.pth
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

import owod.register  # noqa: F401,E402
from src.core import YAMLConfig  # noqa: E402


@torch.no_grad()
def bench(model, post, x, sizes, amp, warmup, iters):
    for _ in range(warmup):
        with torch.autocast('cuda', enabled=amp):
            post(model(x), sizes)
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        with torch.autocast('cuda', enabled=amp):
            post(model(x), sizes)
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-c', '--config', required=True)
    ap.add_argument('-r', '--resume', required=True)
    ap.add_argument('--warmup', type=int, default=50)
    ap.add_argument('--iters', type=int, default=500)
    ap.add_argument('--pp', nargs='*', default=[], help='OWPostProcessor overrides, e.g. unk_self_iou=0.7 nms_iou=0.5')
    args = ap.parse_args()
    pp = {k: float(v) for k, v in (a.split('=') for a in args.pp)}
    cfg = YAMLConfig(args.config, OWPostProcessor=pp) if pp else YAMLConfig(args.config)
    model = cfg.model
    model.load_state_dict(torch.load(args.resume, map_location='cpu')['ema']['module'])
    model = model.cuda().eval()
    post = cfg.postprocessor.cuda().eval()
    torch.manual_seed(0)
    x = torch.rand(1, 3, 640, 640, device='cuda')
    sizes = torch.tensor([[640, 480]], device='cuda')
    for amp in (False, True):
        dt = bench(model, post, x, sizes, amp, args.warmup, args.iters)
        print(f'{os.path.basename(args.config)} amp={amp}: {dt * 1000:.2f} ms/img  {1 / dt:.1f} FPS', flush=True)


if __name__ == '__main__':
    main()
