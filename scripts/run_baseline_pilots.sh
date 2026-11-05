#!/usr/bin/env bash
# 2-epoch pilots of the RT-DETRv2 baselines (docs/BASELINES_RTDETR.md), run after the OW-DETR pilot.
cd /data1/AnoDet/owod
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
while pgrep -f "output_dir=./outputs/pilot_owdetr_rt" > /dev/null; do sleep 30; done
for c in cat_rt randbox_rt orthogonaldet_rt; do
  echo "$(date '+%F %T') start $c"
  extra=""
  [ $c = cat_rt ] && extra="OWCriterion.attn_pseudo_start=1"
  uv run python tools/train.py -c configs/baselines/$c.yml -u epoches=2 eval_freq=1 output_dir=./outputs/pilot_$c $extra \
    > outputs/logs/pilot_$c.log 2>&1
  echo "$(date '+%F %T') done $c (exit $?)"
done
echo "$(date '+%F %T') all done"
