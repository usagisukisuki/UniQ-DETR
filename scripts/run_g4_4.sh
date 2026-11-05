#!/usr/bin/env bash
# G4-4: G4-1 trained with 600 queries (analysis/G4_PLAN.md), 50 epochs, then the G3 sweep, the 0a score
# formula sweep and the same analysis as G3.
cd /data1/AnoDet/owod
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
RESUME=${RESUME:-}
C="-c configs/g4_4.yml -r outputs/g4_4/best.pth"
echo "$(date '+%F %T') start"
uv run python tools/train.py -c configs/g4_4.yml ${RESUME:+-r $RESUME} >> outputs/logs/g4_4.log 2>&1
rc=$?
echo "$(date '+%F %T') trained (exit $rc)"
[ $rc -eq 0 ] || exit $rc
uv run python tools/sweep_pp.py $C --out outputs/g4_4/sweep.json \
  --grid unk_r_gamma=1,0.5,0.25 unk_weight=0.5,1,1.5,2,3,4,6,8,12 > outputs/logs/sweep_g4_4.log 2>&1
uv run python tools/sweep_pp.py $C --out outputs/g4_4/sweep_exp.json \
  --grid unk_o_exp=0.5,1 unk_r_gamma=0.25,0.5 unk_q_exp=1,2 unk_weight=2,3,4,6,8 > outputs/logs/sweep_g4_4_exp.log 2>&1
for w in 1 1.5 2; do
  uv run python tools/diag_unk_output.py $C --w $w >> outputs/logs/diag_unk_output_g4_4.log 2>&1
done
uv run python tools/diag_unk_dets.py $C > outputs/logs/diag_unk_dets_g4_4.log 2>&1
echo "$(date '+%F %T') all done"
