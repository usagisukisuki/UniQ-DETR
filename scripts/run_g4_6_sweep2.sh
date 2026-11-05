#!/usr/bin/env bash
# G4-6: wider score sweep (r up to 2, w up to 64, area 0.125..0.375) after r moved to gamma 1 (analysis/G4_PLAN.md 6.6).
cd /data1/AnoDet/owod
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${GPU:-0}
echo "$(date '+%F %T') start"
for c in g4_6_nq900 g4_6; do
  uv run python tools/sweep_pp.py -c configs/$c.yml -r outputs/g4_6/best.pth --out outputs/g4_6/sweep_area2_$c.json \
    --grid unk_area_exp=0.125,0.25,0.375 unk_r_gamma=0.75,1,1.5,2 unk_q_exp=0.5,1,2 unk_weight=8,12,16,24,32,48,64,96 \
    > outputs/logs/sweep_area2_$c.log 2>&1
  echo "$(date '+%F %T') $c swept (exit $?)"
done
uv run python tools/diag_unk_miss.py -c configs/g4_6_nq900.yml -r outputs/g4_6/best.pth \
  --pp unk_area_exp=0.25 unk_r_gamma=1 unk_q_exp=1 unk_weight=12 > outputs/logs/diag_unk_miss_g4_6_r1.log 2>&1
echo "$(date '+%F %T') all done"
