#!/bin/bash
# D (analysis/UNK_MISS_ANALYSIS.md 5.2): reserve query slots for the unknown column at inference (G4-8 weights)
set -e
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=0
for s in "all 100" "all 200" "all 300" "known 150" "known 300"; do
  set -- $s
  tag=${1}_$2
  echo "== $tag"
  uv run python tools/diag_unk_funnel.py -c configs/uniq_detr.yml -r outputs/g4_8/best.pth --pp unk_weight=12 \
      --dec unk_select=$2 unk_select_base=$1 2>&1 | tr '\r' '\n' | grep -v "it/s\]\|s/it\]" > outputs/logs/diag_unk_funnel_d_$tag.log
  uv run python tools/sweep_pp.py -c configs/uniq_detr.yml -r outputs/g4_8/best.pth --full --floor 63.5 \
      --dec unk_select=$2 unk_select_base=$1 --grid unk_weight=8,12,16,24 --out outputs/g4_8/sweep_d_$tag.json \
      2>&1 | tr '\r' '\n' | grep -v "it/s\]\|s/it\]" > outputs/logs/sweep_d_$tag.log
done
echo all done
