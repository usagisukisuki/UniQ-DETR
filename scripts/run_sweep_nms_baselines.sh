#!/usr/bin/env bash
# NMS (IoU 0.5 / 0.7) on every baseline, same split-half w selection as docs/BASELINES_RESULTS.md (§9.2).
cd /data1/AnoDet/owod
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
run() {  # name config ckpt w-grid
  echo "$(date '+%F %T') start $1"
  uv run python tools/sweep_pp.py -c $2 -r $3 --out $(dirname $3)/sweep_nms.json --grid nms_iou=0.5,0.7 unk_weight=$4 \
    > outputs/logs/sweep_nms_$1.log 2>&1
  echo "$(date '+%F %T') done $1 (exit $?)"
}
run prob_rt configs/b_prob_rt.yml outputs/e2_obj/best.pth 0.5,0.6,0.7,0.8,1,1.2,1.5,2
run b_owdetr_rt configs/baselines/owdetr_rt.yml outputs/b_owdetr_rt/best.pth 8,16,32,64,128,256
run b_cat_rt configs/baselines/cat_rt.yml outputs/b_cat_rt/best.pth 1.5,2,2.5,3,4,6,8
run b_orthogonaldet_rt configs/baselines/orthogonaldet_rt.yml outputs/b_orthogonaldet_rt/best.pth 2,4,8,16,32,64
run b_randbox_rt configs/baselines/randbox_rt.yml outputs/b_randbox_rt/best.pth 64,128,256,512,1024,2048,4096
echo "$(date '+%F %T') all done"
