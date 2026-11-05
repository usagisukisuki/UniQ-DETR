"""Open-world evaluator for M-OWODB.

Faithful port of PROB `datasets/open_world_eval.py` (itself from ORE):
  * VOC07 11-point AP at IoU>0.5 (strict `>`), VOC "+1" pixel convention,
    detection lines rounded like PROB (score %.3f, box %.1f, xmin/ymin +1).
  * Known mAP   = mean AP50 over the known classes.
  * U-Recall    = recall of the `unknown` class (all non-known GT objects).
  * WI          = Wilderness Impact at known-recall 0.8.
  * A-OSE       = # known-class detections overlapping an unknown GT (IoU>0.5).

Predictions use label in [0, num_known) for known classes and `num_known` for unknown.
GT comes from `test.json` written by scripts/prepare_mowodb.py (`bbox_voc`, `difficult`).
"""

import json
from collections import defaultdict

import numpy as np
import torch

from ..data.classes import ALL_CLASSES, UNKNOWN_NAME


def voc_ap(rec, prec, use_07_metric=True):
    if use_07_metric:
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            p = 0 if np.sum(rec >= t) == 0 else np.max(prec[rec >= t])
            ap += p / 11.
        return ap
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    i = np.where(mrec[1:] != mrec[:-1])[0]
    return np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])


def _iou_voc(gt, bb):
    ixmin = np.maximum(gt[:, 0], bb[0])
    iymin = np.maximum(gt[:, 1], bb[1])
    ixmax = np.minimum(gt[:, 2], bb[2])
    iymax = np.minimum(gt[:, 3], bb[3])
    iw = np.maximum(ixmax - ixmin + 1., 0.)
    ih = np.maximum(iymax - iymin + 1., 0.)
    inters = iw * ih
    uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
           (gt[:, 2] - gt[:, 0] + 1.) * (gt[:, 3] - gt[:, 1] + 1.) - inters)
    return inters / uni


class OWODEvaluator:
    """Accumulates predictions and computes OWOD metrics against test.json GT."""

    iou_types = ('bbox',)

    def __init__(self, gt_json, num_known=20, ovthresh=0.5, use_07_metric=True):
        self.num_known = num_known
        self.ovthresh = ovthresh
        self.use_07_metric = use_07_metric
        self.class_names = list(ALL_CLASSES[:num_known]) + [UNKNOWN_NAME]
        gt = json.load(open(gt_json)) if isinstance(gt_json, str) else gt_json
        assert gt['info']['split'] == 'test'
        self.image_ids = [img['id'] for img in gt['images']]
        # owod_all_task_test.txt lists 1,127 images twice. PROB/ORE voc_eval keys GT by image *name*
        # (one GT set per name, so a duplicate listing's detections hit already-matched GT -> FP) but
        # counts npos once per *listing*. Reproduce that: match on a per-name key, npos over listings.
        # Without this, Known AP50 / U-Recall are ~6pt higher than PROB's evaluator on identical detections.
        self.key = {img['id']: img.get('owod_name', img['id']) for img in gt['images']}
        first = {}
        for img in gt['images']:
            first.setdefault(self.key[img['id']], img['id'])
        # gt_by_cls[cls_idx][key] = (boxes[N,4] float, difficult[N] bool); npos[cls] over all listings
        tmp = defaultdict(lambda: defaultdict(lambda: ([], [])))
        self.npos = defaultdict(int)
        for a in gt['annotations']:
            c = a['category_id'] if a['category_id'] < num_known else num_known
            self.npos[c] += int(not a['difficult'])
            k = self.key[a['image_id']]
            if first[k] != a['image_id']:
                continue
            b, d = tmp[c][k]
            b.append(a['bbox_voc'])
            d.append(bool(a['difficult']))
        self.gt = {}
        for c in range(num_known + 1):
            self.gt[c] = {k: (np.asarray(b, dtype=np.float64).reshape(-1, 4), np.asarray(d, dtype=bool))
                          for k, (b, d) in tmp[c].items()}
        self.cleanup()

    def cleanup(self):
        self.preds = defaultdict(list)  # cls -> list of (image_id, score, x1, y1, x2, y2)

    def update(self, predictions):
        """predictions: {image_id: {'boxes': [K,4] xyxy pixels, 'labels': [K], 'scores': [K]}}"""
        for img_id, p in predictions.items():
            boxes = p['boxes'].detach().float().cpu().numpy()
            labels = p['labels'].detach().cpu().numpy()
            scores = p['scores'].detach().float().cpu().numpy()
            img_id = self.key[int(img_id)]
            for (x1, y1, x2, y2), c, s in zip(boxes, labels, scores):
                # PROB rounding: score %.3f, box %.1f after +1 on xmin/ymin
                # float() first: rounding a numpy float32 keeps float32 (4.4 -> 4.4000001), which flips IoU>0.5 ties
                self.preds[int(c)].append((img_id, round(float(s), 3), round(float(x1) + 1., 1), round(float(y1) + 1., 1),
                                           round(float(x2), 1), round(float(y2), 1)))

    def synchronize_between_processes(self):
        from ..misc import all_gather_object
        merged = defaultdict(list)
        for d in all_gather_object(dict(self.preds)):
            for k, v in d.items():
                merged[k].extend(v)
        self.preds = merged

    def _eval_class(self, c):
        gt = self.gt[c]
        npos = self.npos[c]
        dets = self.preds.get(c, [])
        if len(dets) == 0:
            return dict(rec=np.zeros(0), prec=np.zeros(0), ap=0., npos=npos, ose=0.,
                        tp_fp=np.zeros(0), fp_os=np.zeros(0))
        img = np.array([d[0] for d in dets])
        conf = np.array([d[1] for d in dets])
        BB = np.array([d[2:] for d in dets], dtype=np.float64)
        # Scores are rounded to 3 decimals (as in PROB), so ties are massive. PROB uses the default
        # np.argsort, whose tie order depends on the numpy version/CPU SIMD path (numpy 1.26 vs 2.x give
        # U-AP50 1.71 vs 1.49 on the same PROB detections). Use a stable sort so results are reproducible;
        # ties then keep insertion (= image) order. Known AP50 moves by ~0.01pt; U-Recall/A-OSE unaffected.
        order = np.argsort(-conf, kind='stable')
        img, BB = img[order], BB[order]
        nd = len(img)
        tp = np.zeros(nd)
        fp = np.zeros(nd)
        used = {iid: np.zeros(len(b), dtype=bool) for iid, (b, _) in gt.items()}
        for k in range(nd):
            g = gt.get(img[k])
            if g is None or len(g[0]) == 0:
                fp[k] = 1.
                continue
            ov = _iou_voc(g[0], BB[k])
            j = int(np.argmax(ov))
            if ov[j] > self.ovthresh:
                if not g[1][j]:
                    if not used[img[k]][j]:
                        tp[k] = 1.
                        used[img[k]][j] = True
                    else:
                        fp[k] = 1.
            else:
                fp[k] = 1.
        fp, tp = np.cumsum(fp), np.cumsum(tp)
        rec = tp / float(max(npos, 1)) if npos > 0 else tp * 0.
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = voc_ap(rec, prec, self.use_07_metric)

        if c == self.num_known:
            return dict(rec=rec, prec=prec, ap=ap, npos=npos, ose=0., tp_fp=None, fp_os=None)

        # open-set errors: detections of this known class overlapping an unknown GT
        unk = self.gt[self.num_known]
        is_unk = np.zeros(nd)
        for k in range(nd):
            g = unk.get(img[k])
            if g is None or len(g[0]) == 0:
                continue
            if _iou_voc(g[0], BB[k]).max() > self.ovthresh:
                is_unk[k] = 1.
        return dict(rec=rec, prec=prec, ap=ap, npos=npos, ose=float(is_unk.sum()),
                    tp_fp=tp + fp, fp_os=np.cumsum(is_unk))

    def accumulate(self):
        self.results = {c: self._eval_class(c) for c in range(self.num_known + 1)}

    def _wi(self, recall_level=0.8):
        tp_fps, fps = [], []
        for c in range(self.num_known):
            r = self.results[c]
            if len(r['rec']) == 0:
                continue
            idx = int(np.argmin(np.abs(r['rec'] - recall_level)))
            tp_fps.append(r['tp_fp'][idx])
            fps.append(r['fp_os'][idx])
        return float(np.mean(fps) / np.mean(tp_fps)) if tp_fps else 0.

    def summarize(self, verbose=True):
        R = self.results
        aps = np.array([R[c]['ap'] * 100 for c in range(self.num_known)])
        last = lambda a: float(a[-1] * 100) if len(a) else 0.
        unk = R[self.num_known]
        stats = {
            'K_AP50': float(aps.mean()),
            'K_R50': float(np.mean([last(R[c]['rec']) for c in range(self.num_known)])),
            'K_P50': float(np.mean([last(R[c]['prec']) for c in range(self.num_known)])),
            'U_R50': last(unk['rec']),
            'U_P50': last(unk['prec']),
            'U_AP50': float(unk['ap'] * 100),
            'WI': self._wi(0.8),
            'A_OSE': float(sum(R[c]['ose'] for c in range(self.num_known))),
            'num_unk_gt': int(unk['npos']),
            'AP50_per_class': {self.class_names[c]: float(aps[c]) for c in range(self.num_known)},
        }
        # RT-DETR solver compatibility: stats[0] is used for best-checkpoint selection
        self.stats = stats
        if verbose:
            print('[OWOD] Known AP50 {K_AP50:.2f} | U-Recall {U_R50:.2f} | WI {WI:.4f} | '
                  'A-OSE {A_OSE:.0f} | U-AP50 {U_AP50:.2f} | #unk GT {num_unk_gt}'.format(**stats))
        return stats
