"""Open-world evaluator for M-OWODB.

Faithful port of PROB `datasets/open_world_eval.py` (itself from ORE):
  * VOC07 11-point AP at IoU>0.5 (strict `>`), VOC "+1" pixel convention,
    detection lines rounded like PROB (score %.3f, box %.1f, xmin/ymin +1).
  * Known mAP   = mean AP50 over the known classes; with prev_known > 0 (Tasks 2-4) also over the previously
    known classes [0, prev_known) and the current ones [prev_known, num_known) (PROB's 'Prev' / 'Cur').
  * U-Recall    = recall of the `unknown` class (all non-known GT objects).
  * WI          = Wilderness Impact at known-recall 0.8.
  * A-OSE       = # known-class detections overlapping an unknown GT (IoU>0.5).
Added after Hosoya et al., "Rethinking Open-Set Object Detection" (IJCV 2025), which scores known and unknown
detection by AP over IoU 0.50:0.95 and notes that A-OSE / WI miss the known -> unknown errors:
  * K_AP / U_AP = Known / unknown AP averaged over IoU 0.50, 0.55, ..., 0.95 with COCO's 101-point AP (same matching as
                  AP50); K_AP75 / U_AP75 at IoU 0.75.
  * K2U         = # known GT objects (non-difficult) with no detection of their own class (IoU>0.5) in the output but
                  an unknown detection (IoU>0.5): known objects reported only as unknown (A-OSE's reverse, per GT, so
                  it neither grows with the number of low-score unknown detections nor depends on how the known and
                  unknown scores compare); K2U_rate = K2U / # known GT.

Predictions use label in [0, num_known) for known classes and `num_known` for unknown.
GT comes from `test.json` written by scripts/prepare_mowodb.py (`bbox_voc`, `difficult`).
"""

import json
from collections import defaultdict

import numpy as np
import torch

from ..data.classes import ALL_CLASSES, UNKNOWN_NAME


IOU_RANGE = np.linspace(0.5, 0.95, 10)   # K_AP / U_AP (COCO's thresholds)
REC_PTS = np.linspace(0., 1., 101)


def coco_ap(rec, prec):
    """COCO's 101-point interpolated AP (used for K_AP / U_AP). VOC07's 11-point rule gives 1/11 to any class whose
    top detection is correct, which dominated the low-recall unknown class."""
    if rec.size == 0:
        return 0.
    mpre = np.maximum.accumulate(prec[::-1])[::-1]
    i = np.searchsorted(rec, REC_PTS, side='left')
    return float(np.where(i < rec.size, mpre[np.minimum(i, rec.size - 1)], 0.).mean())


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

    def __init__(self, gt_json, num_known=20, ovthresh=0.5, use_07_metric=True, prev_known=0):
        self.num_known = num_known
        self.prev_known = prev_known
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
        # non-difficult known GT boxes of each image, for K2U
        kb = defaultdict(list)
        for c in range(num_known):
            for k, (b, d) in self.gt[c].items():
                kb[k].append(b[~d])
        self.known_gt = {k: np.concatenate(v) for k, v in kb.items()}
        self.cleanup()

    def cleanup(self):
        self.preds = defaultdict(list)  # cls -> list of (image_id, score, x1, y1, x2, y2)
        self.pred_ids = defaultdict(list)  # cls -> dataset image id of each prediction (multi-GPU merge)

    def update(self, predictions):
        """predictions: {image_id: {'boxes': [K,4] xyxy pixels, 'labels': [K], 'scores': [K]}}"""
        for img_id, p in predictions.items():
            boxes = p['boxes'].detach().float().cpu().numpy()
            labels = p['labels'].detach().cpu().numpy()
            scores = p['scores'].detach().float().cpu().numpy()
            iid, img_id = int(img_id), self.key[int(img_id)]
            for (x1, y1, x2, y2), c, s in zip(boxes, labels, scores):
                self.pred_ids[int(c)].append(iid)
                # PROB rounding: score %.3f, box %.1f after +1 on xmin/ymin
                # float() first: rounding a numpy float32 keeps float32 (4.4 -> 4.4000001), which flips IoU>0.5 ties
                self.preds[int(c)].append((img_id, round(float(s), 3), round(float(x1) + 1., 1), round(float(y1) + 1., 1),
                                           round(float(x2), 1), round(float(y2), 1)))

    def synchronize_between_processes(self):
        from ..misc import all_gather_object
        parts = all_gather_object((dict(self.preds), dict(self.pred_ids)))
        if len(parts) == 1:
            return
        # multi-GPU: the DistributedSampler interleaves the images over the ranks and pads the last batch with repeats.
        # Keep each image (dataset id; duplicate listings share the name key but not the id) from the first rank that
        # has it, and restore the single-GPU (dataset) order, which the stable score sort uses to break ties.
        pos = {i: n for n, i in enumerate(self.image_ids)}
        owner = {}
        for r, (_, ids) in enumerate(parts):
            for v in ids.values():
                for i in v:
                    owner.setdefault(i, r)
        merged = defaultdict(list)
        for r, (preds, ids) in enumerate(parts):
            for c, v in preds.items():
                merged[c].extend((pos[i], det) for i, det in zip(ids[c], v) if owner[i] == r)
        self.preds = defaultdict(list, {c: [det for _, det in sorted(v, key=lambda x: x[0])] for c, v in merged.items()})
        self.pred_ids = defaultdict(list)

    def _eval_class(self, c):
        gt = self.gt[c]
        npos = self.npos[c]
        dets = self.preds.get(c, [])
        if len(dets) == 0:
            return dict(rec=np.zeros(0), prec=np.zeros(0), ap=0., aps=[0.] * len(IOU_RANGE), npos=npos, ose=0.,
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
        # best-overlapping GT of each detection (independent of the IoU threshold); gid numbers the GT over images
        ovmax = np.full(nd, -1.)
        gid = np.full(nd, -1, dtype=np.int64)
        diff = np.zeros(nd, dtype=bool)
        offset, n_gt = {}, 0
        for iid, (b, _) in gt.items():
            offset[iid] = n_gt
            n_gt += len(b)
        for k in range(nd):
            g = gt.get(img[k])
            if g is None or len(g[0]) == 0:
                continue
            ov = _iou_voc(g[0], BB[k])
            j = int(np.argmax(ov))
            ovmax[k], gid[k], diff[k] = ov[j], offset[img[k]] + j, g[1][j]

        def pr(thr):
            # VOC greedy matching: the first (highest-score) detection on a non-difficult GT is a TP, later ones FP;
            # a detection on a difficult GT is neither; below the threshold it is an FP
            hit = ovmax > thr
            cand = np.nonzero(hit & ~diff)[0]
            tp = np.zeros(nd)
            tp[cand[np.unique(gid[cand], return_index=True)[1]]] = 1.
            fp = (~hit).astype(np.float64)
            fp[cand] = 1. - tp[cand]
            fp, tp = np.cumsum(fp), np.cumsum(tp)
            rec = tp / float(max(npos, 1)) if npos > 0 else tp * 0.
            prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
            return rec, prec, tp, fp

        rec, prec, tp, fp = pr(self.ovthresh)
        ap = voc_ap(rec, prec, self.use_07_metric)
        aps = [coco_ap(*pr(t)[:2]) for t in IOU_RANGE]

        if c == self.num_known:
            return dict(rec=rec, prec=prec, ap=ap, aps=aps, npos=npos, ose=0., tp_fp=None, fp_os=None)

        # open-set errors: detections of this known class overlapping an unknown GT
        unk = self.gt[self.num_known]
        is_unk = np.zeros(nd)
        for k in range(nd):
            g = unk.get(img[k])
            if g is None or len(g[0]) == 0:
                continue
            if _iou_voc(g[0], BB[k]).max() > self.ovthresh:
                is_unk[k] = 1.
        return dict(rec=rec, prec=prec, ap=ap, aps=aps, npos=npos, ose=float(is_unk.sum()),
                    tp_fp=tp + fp, fp_os=np.cumsum(is_unk))

    def accumulate(self):
        self.results = {c: self._eval_class(c) for c in range(self.num_known + 1)}
        self.k2u = self._k2u()

    def _k2u(self):
        """# known GT not detected as its own class (no detection of that class with IoU > ovthresh in the output) but
        covered by an unknown detection (IoU > ovthresh): known objects reported only as unknown."""
        def by_img(c):
            v = defaultdict(list)
            for d in self.preds.get(c, []):
                v[d[0]].append(d[2:])
            return {k: np.asarray(b, dtype=np.float64) for k, b in v.items()}
        unk = by_img(self.num_known)
        n = 0
        for c in range(self.num_known):
            own = by_img(c)
            for k, (g, diff) in self.gt[c].items():
                u = unk.get(k)
                if u is None:
                    continue
                o = own.get(k)
                for gb in g[~diff]:
                    if o is not None and (_iou_voc(o, gb) > self.ovthresh).any():
                        continue
                    n += int((_iou_voc(u, gb) > self.ovthresh).any())
        return n

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
        # AP over IoU 0.50:0.95 and at 0.75 (index 5 of IOU_RANGE)
        apr = np.array([R[c]['aps'] for c in range(self.num_known)]) * 100      # [num_known, 10]
        stats.update(K_AP=float(apr.mean()), K_AP75=float(apr[:, 5].mean()),
                     U_AP=float(np.mean(unk['aps']) * 100), U_AP75=float(unk['aps'][5] * 100), K2U=self.k2u,
                     K2U_rate=float(self.k2u / max(sum(len(g) for g in self.known_gt.values()), 1) * 100))
        if self.prev_known:
            stats['PK_AP50'] = float(aps[:self.prev_known].mean())
            stats['CK_AP50'] = float(aps[self.prev_known:].mean())
            stats['PK_AP'] = float(apr[:self.prev_known].mean())
            stats['CK_AP'] = float(apr[self.prev_known:].mean())
        # RT-DETR solver compatibility: stats[0] is used for best-checkpoint selection
        self.stats = stats
        if verbose:
            print('[OWOD] Known AP50 {K_AP50:.2f} | U-Recall {U_R50:.2f} | WI {WI:.4f} | '
                  'A-OSE {A_OSE:.0f} | U-AP50 {U_AP50:.2f} | #unk GT {num_unk_gt}'.format(**stats)
                  + (' | Prev {PK_AP50:.2f} | Cur {CK_AP50:.2f}'.format(**stats) if self.prev_known else ''))
            print('[OWOD] Known AP {K_AP:.2f} (AP75 {K_AP75:.2f}) | U-AP {U_AP:.2f} (AP75 {U_AP75:.2f}) | '
                  'K2U {K2U:.0f} ({K2U_rate:.2f}%)'.format(**stats))
        return stats
