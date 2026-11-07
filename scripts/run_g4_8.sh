#!/usr/bin/env bash
# G4-8: G4-6 + the encoder proposals also learn the pseudo-unknown boxes (analysis/G4_PLAN.md 6.6), 50 epochs, then the
# score sweep (900 / 600 queries at inference), the stage-wise coverage and the miss analysis.
cd /data1/AnoDet/owod
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${GPU:-0}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "$(date '+%F %T') start"
uv run python tools/train.py -c configs/g4_8.yml >> outputs/logs/g4_8.log 2>&1
rc=$?
echo "$(date '+%F %T') trained (exit $rc)"
[ $rc -eq 0 ] || exit $rc
for c in g4_8_nq900 g4_8; do
  uv run python tools/sweep_pp.py -c configs/$c.yml -r outputs/g4_8/best.pth --out outputs/g4_8/sweep_area_$c.json \
    --grid unk_area_exp=0.125,0.25 unk_r_gamma=0.75,1,1.5 unk_q_exp=1,2 unk_weight=4,6,8,12,16,24,32 \
    > outputs/logs/sweep_area_$c.log 2>&1
done
uv run python tools/diag_small_cov.py -c configs/g4_8_nq900.yml -r outputs/g4_8/best.pth > outputs/logs/diag_small_cov_g4_8.log 2>&1
uv run python tools/diag_unk_miss.py -c configs/g4_8_nq900.yml -r outputs/g4_8/best.pth \
  --pp unk_area_exp=0.25 unk_r_gamma=1 unk_q_exp=1 unk_weight=12 > outputs/logs/diag_unk_miss_g4_8.log 2>&1
echo "$(date '+%F %T') all done"
