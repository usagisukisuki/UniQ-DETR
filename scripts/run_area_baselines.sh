#!/usr/bin/env bash
# analysis/G4_PLAN.md 6.6: the box-area unknown factor (unk_area_exp) on the baselines, topk only and NMS 0.5 / 0.7,
# same split-half w selection as scripts/run_sweep_nms_baselines.sh (area 0 reproduces those numbers).
cd /data1/AnoDet/owod
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2
run() {  # name config ckpt nms-grid w-grid
  echo "$(date '+%F %T') start $1"
  uv run python tools/sweep_pp.py -c $2 -r $3 --out $(dirname $3)/sweep_area.json \
    --grid nms_iou=$4 unk_area_exp=0,0.25,0.5 unk_weight=$5 > outputs/logs/sweep_area_$1.log 2>&1
  echo "$(date '+%F %T') done $1 (exit $?)"
}
run prob_rt configs/b_prob_rt.yml outputs/e2_obj/best.pth None,0.5,0.7 0.5,0.6,0.8,1,1.2,1.5,2,3,4,6,8
run b_owdetr_rt configs/baselines/owdetr_rt.yml outputs/b_owdetr_rt/best.pth None,0.5,0.7 8,16,32,64,128,256,512,1024
run b_cat_rt configs/baselines/cat_rt.yml outputs/b_cat_rt/best.pth None,0.5,0.7 1.5,2,3,4,6,8,12,16,32
run b_orthogonaldet_rt configs/baselines/orthogonaldet_rt.yml outputs/b_orthogonaldet_rt/best.pth 0.5,0.7 2,4,8,16,32,64,128,256
echo "$(date '+%F %T') all done"
