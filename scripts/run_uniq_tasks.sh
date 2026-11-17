#!/usr/bin/env bash
# UniQ-DETR on M-OWODB Tasks 2-4 (README, PROB protocol), one GPU, each task from the previous task's final model:
#   pseudo-unknown boxes (E2 epoch-2 teacher; on t{t}_train also minus the previous model's known detections)
#   -> classifier grown to 20t known classes -> stage 1 (t{t}_train, 10 epochs) -> stage 2 (t{t}_ft exemplars)
#   -> evaluation with the Task-1 inference settings (w 12; tune / report halves and the w sweep).
#   GPU=0 nohup scripts/run_uniq_tasks.sh > outputs/logs/run_uniq_tasks.out 2>&1 &
#   TASKS="3 4" ...   resumes from a finished earlier task
cd /data1/AnoDet/owod
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${GPU:-0}
set -e
log() { echo "$(date '+%F %T') $*"; }
TEACHER="-c configs/e2_obj.yml -r outputs/e2_pilot_bn/last.pth"
for t in ${TASKS:-2 3 4}; do
  K=$((20 * t)); P=$((20 * (t - 1))); O=outputs/uniq_t$t
  if [ $t -eq 2 ]; then PREV_CFG=configs/uniq_detr.yml; PREV=outputs/g4_8/best.pth
  else PREV_CFG=configs/uniq_detr_t$((t - 1)).yml; PREV=outputs/uniq_t$((t - 1))/ft/last.pth; fi
  mkdir -p $O
  log "task $t: pseudo-unknown boxes"
  [ -f data/pseudo_unk/t${t}_train_e2ep2_cons_k5.json ] || uv run python tools/make_pseudo_unk.py $TEACHER --num-known $K \
    --ann data/mowodb/t${t}_train.json --out-prefix data/pseudo_unk/t${t}_train_e2ep2_cons --ks 5 \
    --exclude-config $PREV_CFG --exclude-ckpt $PREV --exclude-known $P > outputs/logs/pseudo_t${t}_train.log 2>&1
  [ -f data/pseudo_unk/t${t}_ft_e2ep2_cons_k5.json ] || uv run python tools/make_pseudo_unk.py $TEACHER --num-known $K \
    --ann data/mowodb/t${t}_ft.json --out-prefix data/pseudo_unk/t${t}_ft_e2ep2_cons --ks 5 > outputs/logs/pseudo_t${t}_ft.log 2>&1
  log "task $t: grow the classifier $P -> $K"
  uv run python tools/expand_classes.py -c configs/uniq_t${t}_train.yml -r $PREV --old-known $P -o $O/init.pth \
    > outputs/logs/expand_t$t.log 2>&1
  log "task $t: stage 1 (t${t}_train)"
  uv run python tools/train.py -c configs/uniq_t${t}_train.yml -t $O/init.pth >> outputs/logs/uniq_t${t}_train.log 2>&1
  log "task $t: stage 2 (t${t}_ft)"
  uv run python tools/train.py -c configs/uniq_t${t}_ft.yml -t $O/train/last.pth >> outputs/logs/uniq_t${t}_ft.log 2>&1
  log "task $t: evaluation"
  uv run python tools/train.py -c configs/uniq_detr_t$t.yml --test-only -r $O/ft/last.pth > outputs/logs/eval_uniq_t$t.log 2>&1
  uv run python tools/sweep_pp.py -c configs/uniq_detr_t$t.yml -r $O/ft/last.pth --out $O/sweep_w.json \
    --grid unk_weight=4,6,8,12,16,24,32 > outputs/logs/sweep_uniq_t$t.log 2>&1
  grep '\[OWOD\]' outputs/logs/eval_uniq_t$t.log | head -2
done
log "all done"
