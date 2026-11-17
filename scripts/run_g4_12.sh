#!/usr/bin/env bash
# G4-12: G4-8 + dense loss over every encoder proposal (analysis/UNK_MISS_ANALYSIS.md 5.2 A), 50 epochs on GPU0,
# then the w sweep (UniQ inference settings) and the unknown funnel on the same GPU.
#   nohup scripts/run_g4_12.sh > outputs/logs/run_g4_12.out 2>&1 &
cd /data1/AnoDet/owod
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
echo "$(date '+%F %T') start"
uv run python tools/train.py -c configs/g4_12.yml >> outputs/logs/g4_12.log 2>&1
rc=$?
echo "$(date '+%F %T') trained (exit $rc)"
[ $rc -eq 0 ] || exit $rc
uv run python tools/sweep_pp.py -c configs/uniq_detr.yml -r outputs/g4_12/best.pth --full --floor 63.5 \
  --grid unk_weight=6,8,12,16,24 --out outputs/g4_12/sweep_w.json 2>&1 | tr '\r' '\n' | grep -v "it/s\]\|s/it\]" \
  > outputs/logs/sweep_w_g4_12.log
uv run python tools/diag_unk_funnel.py -c configs/uniq_detr.yml -r outputs/g4_12/best.pth --pp unk_weight=12 2>&1 \
  | tr '\r' '\n' | grep -v "it/s\]\|s/it\]" > outputs/logs/diag_unk_funnel_g4_12.log
echo "$(date '+%F %T') all done"
