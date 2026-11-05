"""Re-score PROB's dumped Task-1 detections with owod/eval/owod_eval.py and test.json.

PROB labels: 0..19 known, 80 = unknown (TOWOD class list + 'unknown'). Ours: 0..19 known, 20 = unknown.
  uv run python tools/prob_check/compare_eval.py third_party/prob_eval_out/t1_dets.pt
"""
import json
import os
import sys
from collections import Counter

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from owod.eval.owod_eval import OWODEvaluator  # noqa: E402

K, PROB_UNK = 20, 80
dets = torch.load(sys.argv[1])
gt = json.load(open('data/mowodb/test.json'))
# owod_all_task_test.txt lists 1,127 images twice; test.json keeps each listing as its own image id,
# PROB runs the model once per listing; pair the k-th listing with the k-th detection set.
name2ids = {}
for i in gt['images']:
    name2ids.setdefault(i['owod_name'], []).append(i['id'])
assert set(dets) == set(name2ids), (len(set(dets) ^ set(name2ids)), 'image sets differ')

lab_cnt = Counter()
preds = {}
seen = Counter()
for img in gt['images']:  # listing order == PROB's dataset order (matters for score ties)
    name = img['owod_name']
    assert len(dets[name]) == len(name2ids[name]), name
    p = dets[name][seen[name]]
    seen[name] += 1
    lab = p['labels'].clone()
    lab_cnt.update(lab.tolist())
    keep = (lab < K) | (lab == PROB_UNK)
    lab[lab == PROB_UNK] = K
    preds[img['id']] = dict(boxes=p['boxes'][keep], labels=lab[keep], scores=p['scores'][keep])
print('PROB label histogram (top):', lab_cnt.most_common(5), '| dropped (20..79):',
      sum(v for k, v in lab_cnt.items() if K <= k < PROB_UNK))
ev = OWODEvaluator(gt, num_known=K)
ev.update(preds)
ev.accumulate()
print(ev.summarize())
