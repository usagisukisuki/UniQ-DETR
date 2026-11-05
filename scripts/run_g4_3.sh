#!/usr/bin/env bash
# G4-3: G4-1 + training-only one-to-many branch (analysis/G4_PLAN.md), 50 epochs, then the G3 sweep, the 0a score
# formula sweep and the same analysis as G3.
cd /data1/AnoDet/owod
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
# the first run hit CUDA OOM in epoch 33 (fragmentation: 1.5 GB reserved but unallocated); resumed from its last.pth
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
RESUME=${RESUME:-}
C="-c configs/g4_3.yml -r outputs/g4_3/best.pth"
echo "$(date '+%F %T') start"
uv run python tools/train.py -c configs/g4_3.yml ${RESUME:+-r $RESUME} >> outputs/logs/g4_3.log 2>&1
rc=$?
echo "$(date '+%F %T') trained (exit $rc)"
[ $rc -eq 0 ] || exit $rc
uv run python tools/sweep_pp.py $C --out outputs/g4_3/sweep.json \
  --grid unk_r_gamma=1,0.5,0.25 unk_weight=0.5,1,1.5,2,3,4,6,8,12 > outputs/logs/sweep_g4_3.log 2>&1
uv run python tools/sweep_pp.py $C --out outputs/g4_3/sweep_exp.json \
  --grid unk_o_exp=0.5,1 unk_r_gamma=0.25,0.5 unk_q_exp=1,2 unk_weight=2,3,4,6,8 > outputs/logs/sweep_g4_3_exp.log 2>&1
for w in 1 1.5 2; do
  uv run python tools/diag_unk_output.py $C --w $w >> outputs/logs/diag_unk_output_g4_3.log 2>&1
done
uv run python tools/diag_unk_dets.py $C > outputs/logs/diag_unk_dets_g4_3.log 2>&1
echo "$(date '+%F %T') all done"
