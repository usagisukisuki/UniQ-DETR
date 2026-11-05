#!/usr/bin/env bash
# A: M0 + OW-DETR training signal, 50 epochs, then the common evaluation and the unknown-output analysis.
cd /data1/AnoDet/owod
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
echo "$(date '+%F %T') start"
uv run python tools/train.py -c configs/m0_owdetr.yml > outputs/logs/m0_owdetr.log 2>&1
echo "$(date '+%F %T') trained (exit $?)"
uv run python tools/sweep_pp.py -c configs/m0_owdetr.yml -r outputs/m0_owdetr/best.pth --out outputs/m0_owdetr/sweep.json \
  --grid unk_weight=0.5,0.7,1,1.5,2,3,4 > outputs/logs/sweep_m0_owdetr.log 2>&1
for w in 1 1.5 2; do
  uv run python tools/diag_unk_output.py -c configs/m0_owdetr.yml -r outputs/m0_owdetr/best.pth --w $w >> outputs/logs/diag_unk_output_m0_owdetr.log 2>&1
done
uv run python tools/diag_unk_dets.py -c configs/m0_owdetr.yml -r outputs/m0_owdetr/best.pth > outputs/logs/diag_unk_dets_m0_owdetr.log 2>&1
echo "$(date '+%F %T') all done"
