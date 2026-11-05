#!/usr/bin/env bash
# F0 -> F2 -> F1 fine-tunes from E2 (docs/NEXT_STEPS.md), each followed by the unk_weight sweep.
cd /data1/AnoDet/owod
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
for c in f0_ft f2_dupneg f1_quality; do
  echo "$(date '+%F %T') start $c"
  uv run python tools/train.py -c configs/$c.yml -t outputs/e2_obj/best.pth > outputs/logs/$c.log 2>&1
  echo "$(date '+%F %T') trained $c (exit $?)"
  uv run python tools/sweep_unk.py -c configs/$c.yml -r outputs/$c/best.pth > outputs/logs/sweep_unk_$c.log 2>&1
  if [ $c = f1_quality ]; then
    for u in learned su_q; do
      uv run python tools/sweep_unk.py -c configs/$c.yml -r outputs/$c/best.pth --unk-score $u \
        --out outputs/$c/sweep_unk_$u.json > outputs/logs/sweep_unk_${c}_$u.log 2>&1
    done
  fi
  echo "$(date '+%F %T') swept $c"
done
echo "$(date '+%F %T') all done"
