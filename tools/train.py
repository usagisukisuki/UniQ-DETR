"""Train / evaluate open-world RT-DETRv2 on M-OWODB.

  uv run python tools/train.py -c configs/e0_closed.yml [--test-only -r ckpt.pth] [-u key=value ...]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import owod.register  # noqa: F401,E402
from src.core import YAMLConfig, yaml_utils  # noqa: E402
from src.misc import dist_utils  # noqa: E402

from owod.solver.ow_solver import OWDetSolver  # noqa: E402


def main(args):
    dist_utils.setup_distributed(args.print_rank, args.print_method, seed=args.seed)
    assert not all([args.tuning, args.resume]), 'Only support from_scratch or resume or tuning at one time'
    update_dict = yaml_utils.parse_cli(args.update)
    update_dict.update({k: v for k, v in args.__dict__.items() if k not in ['update', ] and v is not None})
    cfg = YAMLConfig(args.config, **update_dict)
    print('cfg: ', cfg.__dict__)
    solver = OWDetSolver(cfg)
    if args.test_only:
        solver.val()
    else:
        solver.fit()
    dist_utils.cleanup()


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument('-c', '--config', type=str, required=True)
    p.add_argument('-r', '--resume', type=str)
    p.add_argument('-t', '--tuning', type=str)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--use-amp', action='store_true')
    p.add_argument('--output-dir', type=str)
    p.add_argument('--summary-dir', type=str)
    p.add_argument('--test-only', action='store_true', default=False)
    p.add_argument('-u', '--update', nargs='+')
    p.add_argument('--print-method', type=str, default='builtin')
    p.add_argument('--print-rank', type=int, default=0)
    p.add_argument('--local-rank', type=int)
    return p


if __name__ == '__main__':
    main(build_parser().parse_args())
