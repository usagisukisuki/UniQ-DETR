"""Run PROB's official evaluation (main_open_world.py --eval) and dump its raw detections.

Must be run with cwd = third_party/PROB and PROB's own venv. Detections (per owod_name: one entry per listing, each
xyxy boxes in PROB's 0-based pixel coords, labels, scores) go to $PROB_DUMP for re-scoring
with owod/eval/owod_eval.py (tools/prob_check/compare_eval.py).
"""
import os
import sys
import runpy

import numpy as np
import torch

np.bool = bool  # PROB uses the removed alias
sys.path.insert(0, os.getcwd())
from datasets.open_world_eval import OWEvaluator  # noqa: E402

_dump = {}
_orig_update = OWEvaluator.update


def update(self, predictions):
    for img_id, p in predictions.items():
        name = self.voc_gt.convert_image_id(int(img_id), to_string=True)
        # duplicate listings are run separately (different batch padding) -> keep every copy, in order
        _dump.setdefault(name, []).append({k: p[k].detach().cpu() for k in ('boxes', 'labels', 'scores')})
    return _orig_update(self, predictions)


OWEvaluator.update = update
_orig_summarize = OWEvaluator.summarize


def summarize(self, *a, **k):
    torch.save(_dump, os.environ['PROB_DUMP'])
    print(f'dumped {len(_dump)} images to', os.environ['PROB_DUMP'])
    return _orig_summarize(self, *a, **k)


OWEvaluator.summarize = summarize
sys.argv = ['main_open_world.py'] + sys.argv[1:]
runpy.run_path('main_open_world.py', run_name='__main__')
