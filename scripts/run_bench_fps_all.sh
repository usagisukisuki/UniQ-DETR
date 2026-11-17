#!/bin/bash
# FPS of the closed-set RT-DETRv2 and every method at its reported Task-1 post-processing setting, same GPU and session (accuracy-vs-FPS plot).
cd "$(dirname "$0")/.." || exit 1
export CUDA_VISIBLE_DEVICES=${GPU:-0}
LOG=outputs/logs/fps_all_settings.log; : > $LOG
b() { echo "== $1" >> $LOG; uv run python tools/bench_fps.py -c $2 -r $3 ${4:+--pp "${@:4}"} >> $LOG 2>&1; }
b warmup configs/uniq_detr.yml outputs/g4_8/best.pth unk_weight=12   # GPU clocks up; not used
b rtdetrv2 configs/e0_closed.yml outputs/e0_closed/best.pth
b uniq_detr configs/uniq_detr.yml outputs/g4_8/best.pth unk_weight=12
b prob_rt_nms configs/b_prob_rt.yml outputs/e2_obj/best.pth nms_iou=0.7 unk_weight=0.7
b prob_rt_topk configs/b_prob_rt.yml outputs/e2_obj/best.pth nms_iou=None unk_weight=0.6
b owdetr_rt_nms configs/baselines/owdetr_rt.yml outputs/b_owdetr_rt/best.pth nms_iou=0.5 unk_weight=128
b owdetr_rt_topk configs/baselines/owdetr_rt.yml outputs/b_owdetr_rt/best.pth nms_iou=None unk_weight=32
b cat_rt_nms configs/baselines/cat_rt.yml outputs/b_cat_rt/best.pth nms_iou=0.7 unk_weight=16
b cat_rt_topk configs/baselines/cat_rt.yml outputs/b_cat_rt/best.pth nms_iou=None unk_weight=2
b randbox_rt configs/baselines/randbox_rt.yml outputs/b_randbox_rt/best.pth nms_iou=0.5 unk_weight=256
b orthogonaldet_rt configs/baselines/orthogonaldet_rt.yml outputs/b_orthogonaldet_rt/best.pth nms_iou=0.5 unk_weight=8
echo "all done" >> $LOG
