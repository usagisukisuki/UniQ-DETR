"""CAT input-driven pseudo-labelling: selective-search proposals for the training images, precomputed offline.

Parameters follow the CAT/SKDF snippet (third_party/baselines_src/SKDF/models/test_selective_search.py:27-60):
selective_search(img, scale=500, sigma=0.9, min_size=200); drop duplicate rects, regions with size < 2000 pixels and
boxes with aspect ratio > 2. Output: {image_id: [[x1, y1, x2, y2], ...]} in original-image pixels.

  uv run python tools/precompute_selective_search.py --ann data/mowodb/t1_train.json --out data/mowodb/t1_train_ss.json -j 30
"""
import argparse
import json
import os
from multiprocessing import Pool

import numpy as np
import selectivesearch
from PIL import Image


def run(item):
    iid, path = item
    img = np.asarray(Image.open(path).convert('RGB'))
    _, regions = selectivesearch.selective_search(img, scale=500, sigma=0.9, min_size=200)
    seen, boxes = set(), []
    for r in regions:
        if r['rect'] in seen:
            continue
        seen.add(r['rect'])
        if r['size'] < 2000:
            continue
        x, y, w, h = r['rect']
        if w == 0 or h == 0 or w / h > 2 or h / w > 2:
            continue
        boxes.append([x, y, x + w, y + h])
    return iid, boxes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ann', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('-j', type=int, default=16)
    ap.add_argument('--limit', type=int, default=None)
    args = ap.parse_args()
    imgs = json.load(open(args.ann))['images'][:args.limit]
    out = {}
    with Pool(args.j) as pool:
        for n, (iid, boxes) in enumerate(pool.imap_unordered(run, [(i['id'], i['file_name']) for i in imgs], chunksize=8)):
            out[str(iid)] = boxes
            if n % 1000 == 0:
                print(f'{n}/{len(imgs)}', flush=True)
    json.dump(out, open(args.out, 'w'))
    nb = [len(v) for v in out.values()]
    print(f'saved {args.out}: {len(out)} images, boxes/image mean {np.mean(nb):.1f} min {min(nb)} max {max(nb)}')


if __name__ == '__main__':
    main()
