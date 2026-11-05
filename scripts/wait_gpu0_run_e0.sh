#!/usr/bin/env bash
# GPU0 の計算プロセスが 0 になる状態が 2 回連続（60s 間隔）で確認できたら E0 を起動する
cd /data1/AnoDet/owod
UUID=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' '$1==0{print $2}')
ok=0
while true; do
  n=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -c "$UUID")
  if [ "$n" -eq 0 ]; then ok=$((ok+1)); else ok=0; fi
  echo "$(date '+%F %T') gpu0_procs=$n ok=$ok"
  [ "$ok" -ge 2 ] && break
  sleep 60
done
echo "$(date '+%F %T') GPU0 free -> start E0"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 uv run python tools/train.py -c configs/e0_closed.yml \
  > outputs/logs/e0_closed.log 2>&1
echo "$(date '+%F %T') E0 exited with code $?"
