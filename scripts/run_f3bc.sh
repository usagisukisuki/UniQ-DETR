#!/usr/bin/env bash
# F3b / F3c fine-tunes from F1-50ep, each followed by the gamma/weight sweep and the unknown-detection breakdown.
cd /data1/AnoDet/owod
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
for c in f3b_heldout10 f3c_top1_dupneg; do
  echo "$(date '+%F %T') start $c"
  uv run python tools/train.py -c configs/$c.yml -t outputs/f1_quality_50ep/best.pth > outputs/logs/$c.log 2>&1
  echo "$(date '+%F %T') trained $c (exit $?)"
  uv run python tools/sweep_pp.py -c configs/$c.yml -r outputs/$c/best.pth --out outputs/$c/sweep.json \
    --grid unk_score=learned_q,learned_q_v,learned_v unk_v_gamma=0.1,0.25,0.5,1 unk_dup_iou=0.7 unk_dup_cls_p=None,0.05 \
    unk_weight=1,2,4,8 > outputs/logs/sweep_$c.log 2>&1
  uv run python tools/diag_unk_dets.py -c configs/$c.yml -r outputs/$c/best.pth > outputs/logs/diag_unk_dets_$c.log 2>&1
  echo "$(date '+%F %T') evaluated $c"
done
echo "$(date '+%F %T') all done"
