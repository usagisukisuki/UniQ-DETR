"""Initialise a Task-t model from the Task-(t-1) one (README, Tasks 2-4): the classification layers grow from
old_known (+ unknown) (+1) to new_known (+ unknown) (+1) rows.

Rows: known classes 0..old_known-1 keep their weights, the rows after them (the 'unknown' row, if the classifier has
one, and the denoising embedding's padding row) move to the end, and the rows of the new classes keep the fresh
initialisation of the new model (RT-DETR's prior bias). Everything else is copied unchanged, so a model whose
classifier does not depend on the number of known classes (OrthogonalDet-RT: 80 slots + unknown) is copied as is.
The output is loaded with tools/train.py -t.

  uv run python tools/expand_classes.py -c configs/uniq_t2_train.yml -r outputs/g4_8/best.pth --old-known 20 \
      -o outputs/uniq_t2/init_from_t1.pth
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

import owod.register  # noqa: F401,E402
from src.core import YAMLConfig  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-c', '--config', required=True, help='config of the new task')
    ap.add_argument('-r', '--resume', required=True, help='checkpoint of the previous task')
    ap.add_argument('--old-known', type=int, required=True)
    ap.add_argument('-o', '--out', required=True)
    args = ap.parse_args()
    cfg = YAMLConfig(args.config)
    new_known = cfg.yaml_cfg['owod_eval']['num_known']
    old_known = args.old_known
    new = cfg.model.state_dict()
    state = torch.load(args.resume, map_location='cpu')
    old = state['ema']['module'] if 'ema' in state else state['model']
    out, grown = {}, []
    for k, v in new.items():
        o = old[k]
        if o.shape == v.shape:
            out[k] = o
            continue
        # classification rows: [known..., (unknown)] (+ the denoising padding row)
        assert o.shape[1:] == v.shape[1:] and v.shape[0] - o.shape[0] == new_known - old_known, (k, o.shape, v.shape)
        w = v.clone()
        w[:old_known] = o[:old_known]
        w[new_known:] = o[old_known:]
        out[k] = w
        grown.append(f'{k} {tuple(o.shape)} -> {tuple(v.shape)}')
    assert set(old) == set(new), set(old) ^ set(new)
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    torch.save({'model': out}, args.out)
    print('\n'.join(grown))
    print(f'saved {args.out}: {len(out)} tensors, {len(grown)} grown ({old_known} -> {new_known} known classes)')


if __name__ == '__main__':
    main()
