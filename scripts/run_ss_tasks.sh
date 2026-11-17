#!/usr/bin/env bash
# CAT-RT selective-search proposals for the Task 2-4 training files (tools/precompute_selective_search.py), CPU only.
cd /data1/AnoDet/owod
for t in 2 3 4; do for k in ft train; do
  f=data/mowodb/t${t}_${k}_ss.json
  [ -f $f ] && continue
  echo "$(date '+%F %T') start $f"
  uv run python tools/precompute_selective_search.py --ann data/mowodb/t${t}_${k}.json --out $f -j ${J:-16}
done; done
echo "$(date '+%F %T') all done"
