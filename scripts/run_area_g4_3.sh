#!/usr/bin/env bash
# analysis/G4_PLAN.md 6.6: G4-3 weights with the box-area unknown score (300 / 900 queries), then RandBox-RT with the area factor.
cd /data1/AnoDet/owod
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
for c in g4_3 g4_3_nq900; do
  echo "$(date '+%F %T') start $c"
  uv run python tools/sweep_pp.py -c configs/$c.yml -r outputs/g4_3/best.pth --out outputs/g4_3/sweep_area_$c.json \
    --grid unk_area_exp=0,0.25 unk_r_gamma=0.5 unk_q_exp=1,2 unk_weight=3,4,6,8,12,16,24 > outputs/logs/sweep_area_$c.log 2>&1
  echo "$(date '+%F %T') done $c (exit $?)"
done
echo "$(date '+%F %T') start b_randbox_rt"
uv run python tools/sweep_pp.py -c configs/baselines/randbox_rt.yml -r outputs/b_randbox_rt/best.pth --out outputs/b_randbox_rt/sweep_area.json \
  --grid nms_iou=0.5,0.7 unk_area_exp=0,0.25,0.5 unk_weight=64,128,256,512,1024,2048,4096,8192,16384 > outputs/logs/sweep_area_b_randbox_rt.log 2>&1
echo "$(date '+%F %T') done b_randbox_rt (exit $?)"
echo "$(date '+%F %T') all done"
