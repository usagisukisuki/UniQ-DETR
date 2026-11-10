#!/usr/bin/env bash
# G4-9: G4-8 + the one-to-many branch (analysis/G4_PLAN.md), 50 epochs, then the score sweep at 600 queries and the
# A-OSE / coverage analyses.
cd /data1/AnoDet/owod
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${GPU:-0}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "$(date '+%F %T') start"
uv run python tools/train.py -c configs/g4_9.yml >> outputs/logs/g4_9.log 2>&1
rc=$?
echo "$(date '+%F %T') trained (exit $rc)"
[ $rc -eq 0 ] || exit $rc
uv run python tools/sweep_pp.py -c configs/g4_9.yml -r outputs/g4_9/best.pth --out outputs/g4_9/sweep_area_g4_9.json \
  --grid unk_area_exp=0.125,0.25 unk_r_gamma=0.75,1,1.5 unk_q_exp=1,2 unk_weight=4,6,8,12,16,24,32 \
  > outputs/logs/sweep_area_g4_9.log 2>&1
uv run python tools/diag_aose.py -c configs/uniq_detr.yml -r outputs/g4_9/best.pth --nq 600 > outputs/logs/diag_aose_g4_9.log 2>&1
uv run python tools/diag_small_cov.py -c configs/uniq_detr.yml -r outputs/g4_9/best.pth --nq 600 > outputs/logs/diag_small_cov_g4_9.log 2>&1
echo "$(date '+%F %T') all done"
