#!/usr/bin/env bash
# F3 (held-out classes as simulated unknowns) and its control F1c, both fine-tuned 10 ep from F1-50ep.
cd /data1/AnoDet/owod
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
for c in f3_heldout f1c_ft; do
  echo "$(date '+%F %T') start $c"
  uv run python tools/train.py -c configs/$c.yml -t outputs/f1_quality_50ep/best.pth > outputs/logs/$c.log 2>&1
  echo "$(date '+%F %T') trained $c (exit $?)"
  if [ $c = f3_heldout ]; then scores=learned_q,learned_q_v,learned_v,su_v,v; else scores=learned_q; fi
  uv run python tools/sweep_pp.py -c configs/$c.yml -r outputs/$c/best.pth --out outputs/$c/sweep.json \
    --grid unk_score=$scores unk_weight=1,0.7,0.5,0.3 > outputs/logs/sweep_$c.log 2>&1
  uv run python tools/diag_unk_dets.py -c configs/$c.yml -r outputs/$c/best.pth > outputs/logs/diag_unk_dets_$c.log 2>&1
  echo "$(date '+%F %T') evaluated $c"
done
echo "$(date '+%F %T') all done"
