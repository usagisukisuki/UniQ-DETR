#!/usr/bin/env bash
# RandBox-RT and OrthogonalDet-RT on M-OWODB Tasks 2-4: sweep the unknown weight w (NMS 0.5) on the whole test set. The
# Task-1 w of README (256 / 8) sends most known detections to unknown on the Task 2-4 models (RandBox Known 35.3 / 29.5,
# OrthogonalDet 35.8 / 32.0, against a closed-set 51.0 / 44.1 and 50.9 / 43.7). Runs after the baseline runs
# (scripts/run_baseline_tasks.sh), one GPU.
#   GPU=0 nohup scripts/sweep_randbox_tasks.sh > outputs/logs/sweep_randbox_tasks.out 2>&1 &
cd /data1/AnoDet/owod
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${GPU:-0}
for m in ${METHODS:-randbox_rt orthogonaldet_rt}; do
  for t in ${TASKS:-2 3 4}; do
    echo "$(date '+%F %T') $m task $t: w sweep"
    uv run python tools/sweep_pp.py -c configs/baselines/tasks/${m}_t${t}_ft.yml -r outputs/${m}_t$t/ft/last.pth \
      --full --out outputs/${m}_t$t/sweep_w_full.json --grid nms_iou=0.5 unk_weight=0,1,2,4,8,16,32,64,128,256 \
      > outputs/logs/${m}_t${t}_sweep_w.log 2>&1
    grep -A12 'per setting' outputs/logs/${m}_t${t}_sweep_w.log
  done
done
echo "$(date '+%F %T') all done"
