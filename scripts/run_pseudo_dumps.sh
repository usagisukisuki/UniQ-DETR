#!/usr/bin/env bash
# analysis/G4_PLAN.md 6.6: per-checkpoint pseudo-unknown candidates on the V1 probe, then tools/sim_pseudo_update.py
# (anchor = the current labels' teacher E2 epoch 2; teachers = the checkpoints of RUN in training order).
cd /data1/AnoDet/owod
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${GPU:-0}
RUN=${RUN:-g4_6}
PROBE=data/v1_probe/coco_train_probe3000.json
D=outputs/pseudo_sim
echo "$(date '+%F %T') start"
[ -f $D/e2ep2.pt ] || uv run python tools/make_pseudo_unk.py -c configs/e2_obj.yml -r outputs/e2_pilot_bn/last.pth --ann $PROBE --ks 50 --dump $D/e2ep2.pt
for e in 0009 0019 0029 0039 0049; do
  [ -f $D/${RUN}_ep$e.pt ] || uv run python tools/make_pseudo_unk.py -c configs/$RUN.yml -r outputs/$RUN/checkpoint$e.pth --num-known 20 --ann $PROBE --ks 50 --dump $D/${RUN}_ep$e.pt
  echo "$(date '+%F %T') ep$e dumped"
done
uv run python tools/sim_pseudo_update.py --ann $PROBE --dumps $D/e2ep2.pt $D/${RUN}_ep00{09,19,29,39,49}.pt > outputs/logs/sim_pseudo_update_$RUN.log 2>&1
echo "$(date '+%F %T') all done"
