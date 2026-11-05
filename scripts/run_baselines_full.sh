#!/usr/bin/env bash
# 50-epoch runs of the RT-DETRv2 baselines (docs/BASELINES_RTDETR.md), each followed by the common evaluation:
# unk_weight selected on the tune half (tools/sweep_pp.py), unknown-detection breakdown, FPS.
cd /data1/AnoDet/owod
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
for c in owdetr_rt cat_rt randbox_rt orthogonaldet_rt; do
  out=outputs/b_${c}
  echo "$(date '+%F %T') start $c"
  uv run python tools/train.py -c configs/baselines/$c.yml > outputs/logs/b_$c.log 2>&1
  echo "$(date '+%F %T') trained $c (exit $?)"
  uv run python tools/sweep_pp.py -c configs/baselines/$c.yml -r $out/best.pth --out $out/sweep.json \
    --grid unk_weight=4,2,1.5,1,0.7,0.5,0.3,0.2,0.1 > outputs/logs/sweep_b_$c.log 2>&1
  uv run python tools/diag_unk_dets.py -c configs/baselines/$c.yml -r $out/best.pth > outputs/logs/diag_unk_dets_b_$c.log 2>&1
  uv run python tools/bench_fps.py -c configs/baselines/$c.yml -r $out/best.pth > outputs/logs/fps_b_$c.log 2>&1
  echo "$(date '+%F %T') evaluated $c"
done
echo "$(date '+%F %T') all done"
