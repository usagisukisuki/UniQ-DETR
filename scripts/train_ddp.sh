#!/usr/bin/env bash
# Multi-GPU (DDP, one node) training with tools/train.py: GPUS=0,2 scripts/train_ddp.sh -c configs/g4_10.yml [args ...]
# total_batch_size stays 16 (split over the GPUs: 8 per GPU on 2), so lr / schedule are unchanged; sync_bn makes the
# BatchNorm statistics (HybridEncoder, objectness o) cover the whole batch of 16 as in single-GPU training.
cd /data1/AnoDet/owod
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${GPUS:?set GPUS, e.g. GPUS=0,2}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
N=$(echo "$GPUS" | tr ',' '\n' | wc -l)
PORT=${PORT:-$((29500 + RANDOM % 1000))}
# tools/train.py keeps only the last -u, so sync_bn=True joins the caller's -u list instead of adding a second one
args=(); has_u=0
for a in "$@"; do
  args+=("$a")
  if [ "$a" = -u ] || [ "$a" = --update ]; then args+=(sync_bn=True); has_u=1; fi
done
[ $has_u -eq 1 ] || args+=(-u sync_bn=True)
exec uv run torchrun --nproc_per_node=$N --master_port=$PORT tools/train.py "${args[@]}"
