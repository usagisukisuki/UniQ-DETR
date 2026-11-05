"""Build M-OWODB (TOWOD split) annotation files in COCO json format.

Outputs (in --out):
  t1_train.json : Task-1 training set. ONLY the 20 known (VOC) classes.
                  category_id == contiguous class index (0..19).
  test.json     : all_task_test (VOC07 test + COCO val2017 subset), ALL 80 classes,
                  category_id == index in ALL_CLASSES (0..79). Each annotation also
                  carries `difficult` and `bbox_voc` (1-based inclusive ints, the
                  exact convention of the ORE/PROB XMLs) for the OWOD evaluator.

Unknown-class GT is written ONLY to test.json. Training code refuses to load any
file whose info.split is not a train split (see owod/data/mowodb.py).
"""

import argparse
import json
import os
import xml.etree.ElementTree as ET
from collections import Counter

from owod.data.classes import ALL_CLASSES, VOC_CLASS_NAMES, to_owod_name


def read_list(p):
    with open(p) as f:
        return [l.strip() for l in f if l.strip()]


def build_train(args, ids):
    src = json.load(open(args.voc_train_json))
    cat_name = {c['id']: to_owod_name(c['name']) for c in src['categories']}
    name2idx = {n: i for i, n in enumerate(VOC_CLASS_NAMES)}
    wanted = set(ids)
    images, id_map = [], {}
    for img in src['images']:
        stem = os.path.splitext(os.path.basename(img['file_name']))[0]
        if stem not in wanted:
            continue
        new_id = len(images)
        id_map[img['id']] = new_id
        images.append({
            'id': new_id, 'file_name': os.path.join(args.voc_img_dir, img['file_name']),
            'width': img['width'], 'height': img['height'], 'owod_name': stem,
        })
    assert len(images) == len(wanted), (len(images), len(wanted))
    anns = []
    for a in src['annotations']:
        if a['image_id'] not in id_map:
            continue
        name = cat_name[a['category_id']]
        assert name in name2idx, f'non-VOC class {name} in VOC train json'
        anns.append({
            'id': len(anns), 'image_id': id_map[a['image_id']], 'category_id': name2idx[name],
            'bbox': a['bbox'], 'area': a['area'], 'iscrowd': 0, 'source': 'gt',
        })
    return {
        'info': {'split': 'train', 'task': 1, 'num_known': 20, 'benchmark': 'M-OWODB'},
        'images': images,
        'annotations': anns,
        'categories': [{'id': i, 'name': n} for i, n in enumerate(VOC_CLASS_NAMES)],
    }


def parse_voc_xml(path):
    root = ET.parse(path).getroot()
    size = root.find('size')
    w, h = int(size.find('width').text), int(size.find('height').text)
    objs = []
    for o in root.findall('object'):
        bb = o.find('bndbox')
        x1, y1, x2, y2 = (int(float(bb.find(k).text)) for k in ('xmin', 'ymin', 'xmax', 'ymax'))
        objs.append({'name': to_owod_name(o.find('name').text.strip()),
                     'difficult': int(o.find('difficult').text), 'bbox_voc': [x1, y1, x2, y2]})
    return w, h, objs


def build_test(args, ids):
    name2idx = {n: i for i, n in enumerate(ALL_CLASSES)}
    coco = json.load(open(args.coco_val_json))
    coco_cat = {c['id']: to_owod_name(c['name']) for c in coco['categories']}
    coco_img = {os.path.splitext(i['file_name'])[0]: i for i in coco['images']}
    coco_anns = {}
    for a in coco['annotations']:
        coco_anns.setdefault(a['image_id'], []).append(a)

    images, anns = [], []
    for stem in ids:
        new_id = len(images)
        if stem in coco_img:  # COCO val2017 image: all 80 classes, difficult=0 (ORE coco2voc)
            info = coco_img[stem]
            w, h = info['width'], info['height']
            fn = os.path.join(args.coco_val_dir, info['file_name'])
            objs = []
            for a in coco_anns.get(info['id'], []):
                x, y, bw, bh = a['bbox']
                objs.append({'name': coco_cat[a['category_id']], 'difficult': 0,
                             'bbox_voc': [int(x + 1.0), int(y + 1.0), int(x + bw + 1.0), int(y + bh + 1.0)]})
            src = 'coco'
        else:  # VOC07 test image: 20 classes only, original difficult flags
            w, h, objs = parse_voc_xml(os.path.join(args.voc07_xml_dir, stem + '.xml'))
            fn = os.path.join(args.voc_img_dir, stem + '.jpg')
            src = 'voc'
        assert os.path.exists(fn), fn
        images.append({'id': new_id, 'file_name': fn, 'width': w, 'height': h,
                       'owod_name': stem, 'source': src})
        for o in objs:
            x1, y1, x2, y2 = o['bbox_voc']
            bw, bh = x2 - x1 + 1, y2 - y1 + 1
            anns.append({
                'id': len(anns), 'image_id': new_id, 'category_id': name2idx[o['name']],
                'bbox': [x1 - 1, y1 - 1, bw, bh], 'area': bw * bh, 'iscrowd': 0,
                'difficult': o['difficult'], 'bbox_voc': o['bbox_voc'],
            })
    return {
        'info': {'split': 'test', 'benchmark': 'M-OWODB'},
        'images': images,
        'annotations': anns,
        'categories': [{'id': i, 'name': n} for i, n in enumerate(ALL_CLASSES)],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--splits', default='third_party/PROB/data/OWOD/ImageSets/TOWOD')
    ap.add_argument('--voc-img-dir', default='/data2/kato/dataset/voc/VOC_0712_converted/JPEGImages')
    ap.add_argument('--voc-train-json', default='/data2/kato/dataset/voc/VOC_0712_converted/voc0712_train_all.json')
    ap.add_argument('--voc07-xml-dir', default='data/raw/voc07test/VOCdevkit/VOC2007/Annotations')
    ap.add_argument('--coco-val-json', default='/data2/kato/dataset/coco/annotations/instances_val2017.json')
    ap.add_argument('--coco-val-dir', default='/data2/kato/dataset/coco/val2017')
    ap.add_argument('--out', default='data/mowodb')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    train = build_train(args, read_list(os.path.join(args.splits, 'owod_t1_train.txt')))
    test = build_test(args, read_list(os.path.join(args.splits, 'owod_all_task_test.txt')))

    for name, d in (('t1_train.json', train), ('test.json', test)):
        with open(os.path.join(args.out, name), 'w') as f:
            json.dump(d, f)
        print(f"{name}: {len(d['images'])} images, {len(d['annotations'])} annotations")

    cnt = Counter(ALL_CLASSES[a['category_id']] for a in test['annotations'] if not a['difficult'])
    n_known = sum(cnt[c] for c in VOC_CLASS_NAMES)
    print(f'test non-difficult objects: known(T1)={n_known} unknown(T1)={sum(cnt.values()) - n_known}')


if __name__ == '__main__':
    main()
