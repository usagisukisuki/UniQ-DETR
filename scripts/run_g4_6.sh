#!/usr/bin/env bash
# G4-6: G4-4 + r trained on the pseudo-unknown boxes (analysis/G4_PLAN.md 6.6), 50 epochs, then the box-area score sweep
# (600 / 900 queries at inference) and the same analysis as G4-4.
cd /data1/AnoDet/owod
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${GPU:-0}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
RESUME=${RESUME:-}
echo "$(date '+%F %T') start"
uv run python tools/train.py -c configs/g4_6.yml ${RESUME:+-r $RESUME} >> outputs/logs/g4_6.log 2>&1
rc=$?
echo "$(date '+%F %T') trained (exit $rc)"
[ $rc -eq 0 ] || exit $rc
for c in g4_6 g4_6_nq900; do
  uv run python tools/sweep_pp.py -c configs/$c.yml -r outputs/g4_6/best.pth --out outputs/g4_6/sweep_area_$c.json \
    --grid unk_area_exp=0,0.125,0.25 unk_r_gamma=0.5,1 unk_q_exp=1,2 unk_weight=2,3,4,6,8,12,16,24 > outputs/logs/sweep_area_$c.log 2>&1
  uv run python tools/diag_unk_output.py -c configs/$c.yml -r outputs/g4_6/best.pth --w 1 >> outputs/logs/diag_unk_output_g4_6.log 2>&1
  uv run python tools/diag_unk_miss.py -c configs/$c.yml -r outputs/g4_6/best.pth \
    --pp unk_area_exp=0.25 unk_r_gamma=0.5 unk_q_exp=1 unk_weight=12 >> outputs/logs/diag_unk_miss_g4_6.log 2>&1
done
uv run python tools/diag_unk_dets.py -c configs/g4_6_nq900.yml -r outputs/g4_6/best.pth > outputs/logs/diag_unk_dets_g4_6.log 2>&1
echo "$(date '+%F %T') all done"
