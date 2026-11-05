#!/usr/bin/env bash
cd /data1/AnoDet/owod
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
echo "$(date '+%F %T') start"
uv run python tools/train.py -c configs/f3c_50ep.yml > outputs/logs/f3c_50ep.log 2>&1
echo "$(date '+%F %T') trained (exit $?)"
uv run python tools/sweep_pp.py -c configs/f3c_50ep.yml -r outputs/f3c_50ep/best.pth --out outputs/f3c_50ep/sweep.json \
  --grid unk_score=learned,learned_q,learned_q_v unk_v_gamma=0.1,0.25,1 unk_dup_iou=None,0.7 unk_dup_cls_p=None,0.05 \
  unk_weight=0.5,0.7,1,2 > outputs/logs/sweep_f3c_50ep.log 2>&1
uv run python tools/diag_unk_dets.py -c configs/f3c_50ep.yml -r outputs/f3c_50ep/best.pth > outputs/logs/diag_unk_dets_f3c_50ep.log 2>&1
echo "$(date '+%F %T') all done"
