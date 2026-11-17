#!/usr/bin/env bash
# Re-evaluate UniQ-DETR and the baselines (Tasks 1-4, whole test set, the reported inference settings) with the metrics
# added after Hosoya et al. (IJCV 2025): Known / unknown AP over IoU 0.50:0.95 and at 0.75, and K2U (unknown detections
# on known GT). Settings: README (Task 1), analysis/TASKS_PLAN.md (Tasks 2-4; RandBox-RT w 32 and OrthogonalDet-RT w 4
# from the re-sweep). Output: <checkpoint dir>/sweep_ap.json, logs outputs/logs/eval_ap_{m}_t{t}.log.
#   GPU=0 nohup scripts/run_eval_ap_range.sh > outputs/logs/run_eval_ap_range.out 2>&1 &
cd /data1/AnoDet/owod
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${GPU:-0}
run() {  # name task config ckpt grid...
  local n=$1 t=$2 c=$3 r=$4; shift 4
  echo "$(date '+%F %T') start $n task $t"
  uv run python tools/sweep_pp.py -c $c -r $r --full --out $(dirname $r)/sweep_ap.json --grid "$@" \
    > outputs/logs/eval_ap_${n}_t$t.log 2>&1 || echo "FAILED $n task $t"
  grep -A6 'per setting' outputs/logs/eval_ap_${n}_t$t.log | grep full
}
B=configs/baselines/tasks
run uniq_detr 1 configs/uniq_detr.yml outputs/g4_8/best.pth unk_weight=12
run prob_rt 1 configs/b_prob_rt.yml outputs/e2_obj/best.pth nms_iou=None,0.7 unk_weight=0.6,0.7
run owdetr_rt 1 configs/baselines/owdetr_rt.yml outputs/b_owdetr_rt/best.pth nms_iou=None,0.5 unk_weight=32,128
run cat_rt 1 configs/baselines/cat_rt.yml outputs/b_cat_rt/best.pth nms_iou=None,0.7 unk_weight=2,16
run randbox_rt 1 configs/baselines/randbox_rt.yml outputs/b_randbox_rt/best.pth nms_iou=0.5 unk_weight=256
run orthogonaldet_rt 1 configs/baselines/orthogonaldet_rt.yml outputs/b_orthogonaldet_rt/best.pth nms_iou=0.5 unk_weight=8
for t in 2 3 4; do
  run uniq_detr $t configs/uniq_detr_t$t.yml outputs/uniq_t$t/ft/last.pth unk_weight=12
  run prob_rt $t $B/prob_rt_t${t}_ft.yml outputs/prob_rt_t$t/ft/last.pth nms_iou=None,0.7 unk_weight=0.6,0.7
  run owdetr_rt $t $B/owdetr_rt_t${t}_ft.yml outputs/owdetr_rt_t$t/ft/last.pth nms_iou=None,0.5 unk_weight=32,128
  run cat_rt $t $B/cat_rt_t${t}_ft.yml outputs/cat_rt_t$t/ft/last.pth nms_iou=None,0.7 unk_weight=2,16
  run randbox_rt $t $B/randbox_rt_t${t}_ft.yml outputs/randbox_rt_t$t/ft/last.pth nms_iou=0.5 unk_weight=32
  run orthogonaldet_rt $t $B/orthogonaldet_rt_t${t}_ft.yml outputs/orthogonaldet_rt_t$t/ft/last.pth nms_iou=0.5 unk_weight=4
done
echo "$(date '+%F %T') all done"
