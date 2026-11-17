"""Build the M-OWODB Task 2-4 training files (COCO json), following PROB / ORE (TOWOD split).

Outputs (in --out), for task t in 2..4 (known classes = ALL_CLASSES[:20 t]):
  t{t}_train.json : owod_t{t}_train images, ONLY the classes introduced in task t (ALL_CLASSES[20(t-1):20t]); earlier
                    classes and unknowns are removed (PROB remove_prev_class_and_unk_instances).
  t{t}_ft.json    : owod_t{t}_ft images (ORE's fixed balanced exemplar set), ALL known classes 0..20t-1
                    (PROB remove_unknown_instances).
category_id == index in ALL_CLASSES. Images come from VOC 07+12 trainval (VOC classes only, as ORE's XMLs) and COCO
train2017 (all 80 classes). The test file is the Task-1 one (data/mowodb/test.json, all 80 classes).

  uv run python -m scripts.prepare_mowodb_tasks
"""

import argparse
import json
import os

from owod.data.classes import ALL_CLASSES, to_owod_name
from scripts.prepare_mowodb import read_list


def load_sources(args):
    """stem -> (image dict, [(category index, bbox, area, iscrowd)])"""
    name2idx = {n: i for i, n in enumerate(ALL_CLASSES)}
    out = {}
    for js, img_dir in ((args.voc_train_json, args.voc_img_dir), (args.coco_train_json, args.coco_train_dir)):
        src = json.load(open(js))
        cat = {c['id']: name2idx[to_owod_name(c['name'])] for c in src['categories']}
        by_img = {}
        for a in src['annotations']:
            by_img.setdefault(a['image_id'], []).append((cat[a['category_id']], a['bbox'], a['area'], a.get('iscrowd', 0)))
        for img in src['images']:
            stem = os.path.splitext(os.path.basename(img['file_name']))[0]
            out[stem] = ({'file_name': os.path.join(img_dir, img['file_name']), 'width': img['width'],
                          'height': img['height'], 'owod_name': stem}, by_img.get(img['id'], []))
    return out


def build(src, ids, keep, task, kind):
    images, anns = [], []
    for stem in ids:
        info, objs = src[stem]
        iid = len(images)
        images.append({'id': iid, **info})
        for c, bbox, area, crowd in objs:
            if c in keep:
                anns.append({'id': len(anns), 'image_id': iid, 'category_id': c, 'bbox': bbox, 'area': area,
                             'iscrowd': crowd, 'source': 'gt'})
    return {
        'info': {'split': 'train', 'task': task, 'kind': kind, 'num_known': 20 * task, 'benchmark': 'M-OWODB'},
        'images': images,
        'annotations': anns,
        'categories': [{'id': i, 'name': n} for i, n in enumerate(ALL_CLASSES[:20 * task])],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--splits', default='third_party/PROB/data/OWOD/ImageSets/TOWOD')
    ap.add_argument('--voc-img-dir', default='/data2/kato/dataset/voc/VOC_0712_converted/JPEGImages')
    ap.add_argument('--voc-train-json', default='/data2/kato/dataset/voc/VOC_0712_converted/voc0712_train_all.json')
    ap.add_argument('--coco-train-json', default='/data2/kato/dataset/coco/annotations/instances_train2017.json')
    ap.add_argument('--coco-train-dir', default='/data2/kato/dataset/coco/train2017')
    ap.add_argument('--out', default='data/mowodb')
    args = ap.parse_args()
    src = load_sources(args)
    for t in (2, 3, 4):
        for kind, keep in (('train', range(20 * (t - 1), 20 * t)), ('ft', range(20 * t))):
            ids = read_list(os.path.join(args.splits, f'owod_t{t}_{kind}.txt'))
            missing = [s for s in ids if s not in src]
            assert not missing, (t, kind, len(missing), missing[:5])
            d = build(src, ids, set(keep), t, kind)
            for img in d['images']:
                assert os.path.exists(img['file_name']), img['file_name']
            with open(os.path.join(args.out, f't{t}_{kind}.json'), 'w') as f:
                json.dump(d, f)
            n_cls = len({a['category_id'] for a in d['annotations']})
            print(f"t{t}_{kind}.json: {len(d['images'])} images, {len(d['annotations'])} annotations, {n_cls} classes")


if __name__ == '__main__':
    main()
