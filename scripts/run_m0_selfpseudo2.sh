#!/usr/bin/env bash
# B2: B + one-to-one unknown assignment (NMS-free), 50 epochs, then evaluation and analysis.
cd /data1/AnoDet/owod
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
echo "$(date '+%F %T') start"
uv run python tools/train.py -c configs/m0_selfpseudo2.yml > outputs/logs/m0_selfpseudo2.log 2>&1
echo "$(date '+%F %T') trained (exit $?)"
uv run python tools/sweep_pp.py -c configs/m0_selfpseudo2.yml -r outputs/m0_selfpseudo2/best.pth --out outputs/m0_selfpseudo2/sweep.json \
  --grid unk_weight=0.5,0.7,1,1.5,2,3,4 > outputs/logs/sweep_m0_selfpseudo2.log 2>&1
for w in 1 1.5 2; do
  uv run python tools/diag_unk_output.py -c configs/m0_selfpseudo2.yml -r outputs/m0_selfpseudo2/best.pth --w $w >> outputs/logs/diag_unk_output_m0_selfpseudo2.log 2>&1
done
uv run python tools/diag_unk_dets.py -c configs/m0_selfpseudo2.yml -r outputs/m0_selfpseudo2/best.pth > outputs/logs/diag_unk_dets_m0_selfpseudo2.log 2>&1
echo "$(date '+%F %T') all done"
