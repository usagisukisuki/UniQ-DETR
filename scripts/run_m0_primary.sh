#!/usr/bin/env bash
# G: M0 + one-to-one primary head (strictly NMS-free unknowns), 50 epochs, then evaluation (no dedup) and analysis.
cd /data1/AnoDet/owod
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
echo "$(date '+%F %T') start"
uv run python tools/train.py -c configs/m0_primary.yml > outputs/logs/m0_primary.log 2>&1
echo "$(date '+%F %T') trained (exit $?)"
uv run python tools/sweep_pp.py -c configs/m0_primary.yml -r outputs/m0_primary/best.pth --out outputs/m0_primary/sweep.json \
  --grid unk_score=learned_qr,learned_q unk_weight=0.3,0.5,0.7,1,1.5,2,3 > outputs/logs/sweep_m0_primary.log 2>&1
for w in 1 1.5 2; do
  uv run python tools/diag_unk_output.py -c configs/m0_primary.yml -r outputs/m0_primary/best.pth --w $w >> outputs/logs/diag_unk_output_m0_primary.log 2>&1
done
uv run python tools/diag_unk_dets.py -c configs/m0_primary.yml -r outputs/m0_primary/best.pth > outputs/logs/diag_unk_dets_m0_primary.log 2>&1
echo "$(date '+%F %T') all done"
