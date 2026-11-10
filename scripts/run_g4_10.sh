#!/usr/bin/env bash
# G4-10: G4-8 + the one-to-many branch with a group distinct from the one-to-one queries (analysis/G4_PLAN.md), 50 epochs
# on 2 GPUs (scripts/train_ddp.sh), then on the first GPU: the score sweep at 600 queries, known duplicates, A-OSE.
#   GPUS=0,2 nohup scripts/run_g4_10.sh > outputs/logs/run_g4_10.out 2>&1 &
cd /data1/AnoDet/owod
GPUS=${GPUS:-0,2}
echo "$(date '+%F %T') start (GPUS=$GPUS)"
GPUS=$GPUS scripts/train_ddp.sh -c configs/g4_10.yml >> outputs/logs/g4_10.log 2>&1
rc=$?
echo "$(date '+%F %T') trained (exit $rc)"
[ $rc -eq 0 ] || exit $rc
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${GPUS%%,*}
uv run python tools/sweep_pp.py -c configs/g4_10.yml -r outputs/g4_10/best.pth --out outputs/g4_10/sweep_area_g4_10.json \
  --grid unk_area_exp=0.125,0.25 unk_r_gamma=0.75,1,1.5 unk_q_exp=1,2 unk_weight=4,6,8,12,16,24,32 \
  > outputs/logs/sweep_area_g4_10.log 2>&1
uv run python tools/diag_known_dups.py -c configs/g4_10.yml -r outputs/g4_10/best.pth > outputs/logs/diag_known_dups_g4_10.log 2>&1
uv run python tools/diag_aose.py -c configs/uniq_detr.yml -r outputs/g4_10/best.pth --nq 600 > outputs/logs/diag_aose_g4_10.log 2>&1
echo "$(date '+%F %T') all done"
