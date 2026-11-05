"""Check OWODEvaluator == PROB voc_eval (Known AP50, U-Recall, A-OSE, WI, U-AP50) on synthetic predictions,
including duplicated test listings (same image name under two ids)."""
import os, sys, json, random, tempfile, types, importlib.util
import numpy as np, torch
import xml.etree.ElementTree as ET
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from owod.eval.owod_eval import OWODEvaluator
from owod.data.classes import ALL_CLASSES

np.bool = bool  # PROB uses the removed alias
# load PROB voc_eval without its package imports
src = open('third_party/PROB/datasets/open_world_eval.py').read().replace('from util.misc import all_gather', '')
prob = types.ModuleType('prob_eval'); exec(compile(src, 'prob_eval', 'exec'), prob.__dict__)
# Our evaluator breaks score ties with a stable sort (PROB's default argsort tie order depends on the
# numpy version); give PROB the same tie-breaking so the comparison is exact.
_argsort = np.argsort
np.argsort = lambda a, *x, **kw: _argsort(a, *x, **{'kind': 'stable', **kw})

random.seed(0); np.random.seed(0); torch.manual_seed(0)
gt = json.load(open('data/mowodb/test.json'))
imgs = [i for i in gt['images'] if i['source'] == 'coco'][:300] + [i for i in gt['images'] if i['source'] == 'voc'][:300]
# owod_all_task_test.txt lists some images twice (separate ids, same owod_name); include such pairs
cnt = {}
for i in gt['images']:
    cnt.setdefault(i['owod_name'], []).append(i)
chosen = {i['owod_name'] for i in imgs}
imgs += [i for v in cnt.values() if len(v) > 1 and v[0]['owod_name'] not in chosen for i in v][:100]
assert len({i['id'] for i in imgs}) == len(imgs)
ids = {i['id'] for i in imgs}
sub = dict(info=gt['info'], images=imgs, categories=gt['categories'],
           annotations=[a for a in gt['annotations'] if a['image_id'] in ids])
tmp = tempfile.mkdtemp()
by_img = {}
for a in sub['annotations']:
    by_img.setdefault(a['image_id'], []).append(a)
xmls = []
for im in imgs:
    root = ET.Element('annotation')
    for a in by_img.get(im['id'], []):
        o = ET.SubElement(root, 'object')
        ET.SubElement(o, 'name').text = ALL_CLASSES[a['category_id']]
        ET.SubElement(o, 'difficult').text = str(a['difficult'])
        bb = ET.SubElement(o, 'bndbox')
        for k, v in zip(('xmin', 'ymin', 'xmax', 'ymax'), a['bbox_voc']):
            ET.SubElement(bb, k).text = str(v)
    p = os.path.join(tmp, im['owod_name'] + '.xml'); ET.ElementTree(root).write(p); xmls.append(p)

K = 20
ev = OWODEvaluator(sub, num_known=K)
preds = {}
for im in imgs:
    boxes, labels, scores = [], [], []
    for a in by_img.get(im['id'], []):
        if random.random() < 0.7:
            x1, y1, x2, y2 = [v - 1 for v in a['bbox_voc']]
            j = np.random.uniform(-0.15, 0.15, 4) * max(x2 - x1, 1)
            boxes.append([x1 + j[0], y1 + j[1], x2 + j[2], y2 + j[3]])
            c = a['category_id'] if a['category_id'] < K else K
            if random.random() < 0.2: c = random.randrange(K + 1)
            labels.append(c); scores.append(random.random())
    for _ in range(30):
        x, y = random.uniform(0, im['width'] - 20), random.uniform(0, im['height'] - 20)
        boxes.append([x, y, x + random.uniform(10, 200), y + random.uniform(10, 200)])
        labels.append(random.randrange(K + 1)); scores.append(random.random() * 0.6)
    preds[im['id']] = dict(boxes=torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
                           labels=torch.tensor(labels), scores=torch.tensor(scores))
ev.update(preds); ev.accumulate(); mine = ev.summarize()

# PROB path
class GT: CLASS_NAMES = list(ALL_CLASSES[:K]) + ['unknown']; annotations = xmls; image_set = [i['owod_name'] for i in imgs]
id2name = {i['id']: i['owod_name'] for i in imgs}
GT.convert_image_id = staticmethod(lambda i, to_string=True: id2name[i])
args = types.SimpleNamespace(PREV_INTRODUCED_CLS=0, CUR_INTRODUCED_CLS=K, num_classes=K + 1)
pe = prob.OWEvaluator(GT(), ('bbox',), args=args)
pe.update(preds)
pe.lines_cls = torch.tensor(pe.lines_cls)
import contextlib, io
with contextlib.redirect_stdout(io.StringIO()):
    pe.accumulate(); ref = pe.summarize()
pairs = [('K_AP50', 'K_AP50'), ('U_R50', 'U_R50'), ('A_OSE', 'AOSA'), ('WI', 'WI'), ('U_AP50', 'U_AP50')]
ok = True
for m, r in pairs:
    a, b = float(mine[m]), float(ref[r])
    flag = abs(a - b) < 1e-6 * max(1, abs(b))
    ok &= flag
    print(f'{m:7s} mine={a:.6f} prob={b:.6f} {"OK" if flag else "MISMATCH"}')
assert ok
print('ALL MATCH')
