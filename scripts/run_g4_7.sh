#!/usr/bin/env bash
# G4-7a / G4-7b: G4-6 with top-10 / top-15 pseudo-unknowns per image (analysis/G4_PLAN.md 6.6), one after the other,
# each followed by the score sweep (600 / 900 queries at inference) and the miss analysis.
cd /data1/AnoDet/owod
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${GPU:-0}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "$(date '+%F %T') start"
[ -f data/pseudo_unk/e2ep2_cons_k15.json ] || uv run python tools/make_pseudo_unk.py -c configs/e2_obj.yml \
  -r outputs/e2_pilot_bn/last.pth --ann data/mowodb/t1_train.json --out-prefix data/pseudo_unk/e2ep2_cons --ks 15 \
  > outputs/logs/pseudo_k15.log 2>&1 || { echo "pseudo k15 failed"; exit 1; }
echo "$(date '+%F %T') pseudo k15 ready"
for r in g4_7a g4_7b; do
  uv run python tools/train.py -c configs/$r.yml >> outputs/logs/$r.log 2>&1
  rc=$?
  echo "$(date '+%F %T') $r trained (exit $rc)"
  [ $rc -eq 0 ] || continue
  for c in ${r}_nq900 $r; do
    uv run python tools/sweep_pp.py -c configs/$c.yml -r outputs/$r/best.pth --out outputs/$r/sweep_area_$c.json \
      --grid unk_area_exp=0.125,0.25 unk_r_gamma=0.75,1,1.5 unk_q_exp=1,2 unk_weight=4,6,8,12,16,24,32 \
      > outputs/logs/sweep_area_$c.log 2>&1
  done
  uv run python tools/diag_unk_output.py -c configs/${r}_nq900.yml -r outputs/$r/best.pth --w 1 > outputs/logs/diag_unk_output_$r.log 2>&1
  uv run python tools/diag_unk_miss.py -c configs/${r}_nq900.yml -r outputs/$r/best.pth \
    --pp unk_area_exp=0.25 unk_r_gamma=1 unk_q_exp=1 unk_weight=12 > outputs/logs/diag_unk_miss_$r.log 2>&1
  echo "$(date '+%F %T') $r analysed"
done
echo "$(date '+%F %T') all done"
