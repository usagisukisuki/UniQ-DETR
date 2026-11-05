#!/usr/bin/env bash
# analysis/G4_PLAN.md 6.6: training-free checks for more U-Recall on GPU0.
#  A: G4-4 weights with 1200 queries + box-area score   B: pseudo-label quality on the V1 probe (k, round-2 teacher)
#  C: why covered unknown objects are missed
cd /data1/AnoDet/owod
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
PP="unk_area_exp=0.25 unk_r_gamma=0.5 unk_q_exp=1"
PROBE=data/v1_probe/coco_train_probe3000.json
echo "$(date '+%F %T') start"
( uv run python tools/sweep_pp.py -c configs/g4_4_nq1200.yml -r outputs/g4_4/best.pth --out outputs/g4_4/sweep_nq1200_area.json \
    --grid unk_area_exp=0.125,0.25 unk_r_gamma=0.5 unk_q_exp=1 unk_weight=4,6,8,12,16,24 > outputs/logs/sweep_g4_4_nq1200_area.log 2>&1
  uv run python tools/diag_unk_output.py -c configs/g4_4_nq1200.yml -r outputs/g4_4/best.pth --w 1 > outputs/logs/diag_unk_output_g4_4_nq1200.log 2>&1
  echo "$(date '+%F %T') A done" ) &
for c in g4_4 g4_4_nq900; do
  uv run python tools/diag_unk_miss.py -c configs/$c.yml -r outputs/g4_4/best.pth --pp $PP unk_weight=12 \
    >> outputs/logs/diag_unk_miss_g4_4.log 2>&1
done
echo "$(date '+%F %T') C done"
L=outputs/logs/pseudo_probe_round2.log
uv run python tools/make_pseudo_unk.py -c configs/e2_obj.yml -r outputs/e2_pilot_bn/last.pth --ann $PROBE --ks 5 10 15 20 >> $L 2>&1
for rk in cons unk cons_unk; do
  uv run python tools/make_pseudo_unk.py -c configs/g4_4.yml -r outputs/g4_4/best.pth --num-known 20 --rank $rk \
    --area-exp 0.25 --pp unk_r_gamma=0.5 --ann $PROBE --ks 5 10 15 20 >> $L 2>&1
done
uv run python tools/make_pseudo_unk.py -c configs/g4_4.yml -r outputs/g4_4/best.pth --num-known 20 --rank unk \
  --area-exp 0 --pp unk_r_gamma=0.5 --ann $PROBE --ks 5 10 15 20 >> $L 2>&1
echo "$(date '+%F %T') B done"
wait
echo "$(date '+%F %T') all done"
