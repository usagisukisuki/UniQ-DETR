#!/usr/bin/env bash
# G3: G + OW-DETR signal for r, 50 epochs, then strict (no-dedup) evaluation and analysis.
cd /data1/AnoDet/owod
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
echo "$(date '+%F %T') start"
uv run python tools/train.py -c configs/m0_primary_owdetr.yml > outputs/logs/m0_primary_owdetr.log 2>&1
echo "$(date '+%F %T') trained (exit $?)"
uv run python tools/sweep_pp.py -c configs/m0_primary_owdetr.yml -r outputs/m0_primary_owdetr/best.pth --out outputs/m0_primary_owdetr/sweep.json \
  --grid unk_r_gamma=1,0.5,0.25 unk_weight=0.5,1,1.5,2,3,4,6,8,12 > outputs/logs/sweep_m0_primary_owdetr.log 2>&1
for w in 1 1.5 2; do
  uv run python tools/diag_unk_output.py -c configs/m0_primary_owdetr.yml -r outputs/m0_primary_owdetr/best.pth --w $w >> outputs/logs/diag_unk_output_m0_primary_owdetr.log 2>&1
done
uv run python tools/diag_unk_dets.py -c configs/m0_primary_owdetr.yml -r outputs/m0_primary_owdetr/best.pth > outputs/logs/diag_unk_dets_m0_primary_owdetr.log 2>&1
echo "$(date '+%F %T') all done"
