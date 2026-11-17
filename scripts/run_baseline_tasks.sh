#!/usr/bin/env bash
# Baselines on M-OWODB Tasks 2-4 (README), same protocol as UniQ-DETR (scripts/run_uniq_tasks.sh), one GPU, one
# method after another; each task starts from the previous task's final model (Task 1: the README checkpoints):
#   classifier grown to 20t known classes -> stage 1 (t{t}_train, 10 epochs) -> stage 2 (t{t}_ft exemplars)
#   -> the Task-1 inference settings of README on the whole test set (tools/sweep_pp.py --full; topk and NMS rows).
# Configs: scripts/make_baseline_task_configs.py. A failing method is skipped, the others go on. Finished stages leave a
# 'done' file and are skipped on a rerun. outputs/{m}_t{t} are symlinks to /data2/kato/owod_outputs (/data1 is full).
#   GPU=0 nohup scripts/run_baseline_tasks.sh > outputs/logs/run_baseline_tasks.out 2>&1 &
#   METHODS="cat_rt" TASKS="3 4" ...   resumes
cd /data1/AnoDet/owod
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${GPU:-0}
log() { echo "$(date '+%F %T') $*"; }
declare -A T1=([prob_rt]=outputs/e2_obj/best.pth [owdetr_rt]=outputs/b_owdetr_rt/best.pth
  [cat_rt]=outputs/b_cat_rt/best.pth [randbox_rt]=outputs/b_randbox_rt/best.pth
  [orthogonaldet_rt]=outputs/b_orthogonaldet_rt/best.pth)
declare -A GRID=([prob_rt]="nms_iou=None,0.7 unk_weight=0.6,0.7" [owdetr_rt]="nms_iou=None,0.5 unk_weight=32,128"
  [cat_rt]="nms_iou=None,0.7 unk_weight=2,16" [randbox_rt]="nms_iou=0.5 unk_weight=256"
  [orthogonaldet_rt]="nms_iou=0.5 unk_weight=8")
task() {  # method t
  local m=$1 t=$2 C=configs/baselines/tasks O=outputs/$1_t$2 L=outputs/logs/$1_t$2
  local prev=${T1[$m]}; [ $t -gt 2 ] && prev=outputs/${m}_t$((t - 1))/ft/last.pth
  mkdir -p $O
  if [ $m = cat_rt ]; then
    until [ -f data/mowodb/t${t}_train_ss.json ] && [ -f data/mowodb/t${t}_ft_ss.json ]; do sleep 60; done
  fi
  if [ ! -f $O/train/done ]; then
    log "$m task $t: grow the classifier"
    uv run python tools/expand_classes.py -c $C/${m}_t${t}_train.yml -r $prev --old-known $((20 * (t - 1))) -o $O/init.pth \
      > ${L}_expand.log 2>&1 || return 1
    log "$m task $t: stage 1"
    uv run python tools/train.py -c $C/${m}_t${t}_train.yml -t $O/init.pth >> ${L}_train.log 2>&1 || return 1
    touch $O/train/done
  fi
  if [ ! -f $O/ft/done ]; then
    log "$m task $t: stage 2"
    uv run python tools/train.py -c $C/${m}_t${t}_ft.yml -t $O/train/last.pth >> ${L}_ft.log 2>&1 || return 1
    touch $O/ft/done
  fi
  log "$m task $t: evaluation"
  uv run python tools/sweep_pp.py -c $C/${m}_t${t}_ft.yml -r $O/ft/last.pth --full --out $O/sweep_full.json \
    --grid ${GRID[$m]} > ${L}_eval.log 2>&1 || return 1
  grep -A5 'per setting' ${L}_eval.log | grep full
}
for m in ${METHODS:-prob_rt owdetr_rt randbox_rt orthogonaldet_rt cat_rt}; do
  for t in ${TASKS:-2 3 4}; do
    task $m $t || { log "$m task $t FAILED, skipping the rest of $m"; break; }
  done
done
log "all done"
