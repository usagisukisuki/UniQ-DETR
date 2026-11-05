#!/usr/bin/env bash
# B: M0 + self-selected pseudo-unknowns, 50 epochs, then the common evaluation and the unknown-output analysis.
cd /data1/AnoDet/owod
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
echo "$(date '+%F %T') start"
uv run python tools/train.py -c configs/m0_selfpseudo.yml > outputs/logs/m0_selfpseudo.log 2>&1
echo "$(date '+%F %T') trained (exit $?)"
uv run python tools/sweep_pp.py -c configs/m0_selfpseudo.yml -r outputs/m0_selfpseudo/best.pth --out outputs/m0_selfpseudo/sweep.json \
  --grid unk_weight=0.5,0.7,1,1.5,2,3,4 > outputs/logs/sweep_m0_selfpseudo.log 2>&1
for w in 1 1.5 2; do
  uv run python tools/diag_unk_output.py -c configs/m0_selfpseudo.yml -r outputs/m0_selfpseudo/best.pth --w $w >> outputs/logs/diag_unk_output_m0_selfpseudo.log 2>&1
done
uv run python tools/diag_unk_dets.py -c configs/m0_selfpseudo.yml -r outputs/m0_selfpseudo/best.pth > outputs/logs/diag_unk_dets_m0_selfpseudo.log 2>&1
echo "$(date '+%F %T') all done"
