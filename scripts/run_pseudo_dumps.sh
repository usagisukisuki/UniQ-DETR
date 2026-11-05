#!/usr/bin/env bash
# analysis/G4_PLAN.md 6.6: per-checkpoint pseudo-unknown candidates on the V1 probe, for tools/sim_pseudo_update.py
cd /data1/AnoDet/owod
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2
PROBE=data/v1_probe/coco_train_probe3000.json
D=outputs/pseudo_sim
uv run python tools/make_pseudo_unk.py -c configs/e2_obj.yml -r outputs/e2_pilot_bn/last.pth --ann $PROBE --ks 50 --dump $D/e2ep2.pt
for e in 0009 0019 0029 0039 0049; do
  uv run python tools/make_pseudo_unk.py -c configs/g4_4.yml -r outputs/g4_4/checkpoint$e.pth --num-known 20 --ann $PROBE --ks 50 --dump $D/g4_4_ep$e.pt
done
echo "$(date '+%F %T') all done"
