#!/usr/bin/env bash
# Task 1 baselines on the WHOLE test set (as PROB), at the settings README picked on the tune half
# (topk and NMS rows); report-half values are printed too, to check they match README.
cd /data1/AnoDet/owod
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${GPU:-0}
run() {  # name config ckpt grid...
  local n=$1 c=$2 r=$3; shift 3
  echo "$(date '+%F %T') start $n"
  uv run python tools/sweep_pp.py -c $c -r $r --full --out $(dirname $r)/sweep_full.json --grid "$@" \
    > outputs/logs/full_test_$n.log 2>&1
  echo "$(date '+%F %T') done $n (exit $?)"
}
run prob_rt configs/b_prob_rt.yml outputs/e2_obj/best.pth nms_iou=None,0.7 unk_weight=0.6,0.7
run owdetr_rt configs/baselines/owdetr_rt.yml outputs/b_owdetr_rt/best.pth nms_iou=None,0.5 unk_weight=32,128
run cat_rt configs/baselines/cat_rt.yml outputs/b_cat_rt/best.pth nms_iou=None,0.7 unk_weight=2,16
run randbox_rt configs/baselines/randbox_rt.yml outputs/b_randbox_rt/best.pth nms_iou=0.5 unk_weight=256
run orthogonaldet_rt configs/baselines/orthogonaldet_rt.yml outputs/b_orthogonaldet_rt/best.pth nms_iou=0.5 unk_weight=8
echo "$(date '+%F %T') all done"
