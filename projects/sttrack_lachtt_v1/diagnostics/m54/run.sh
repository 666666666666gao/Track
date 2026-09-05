#!/usr/bin/env bash
set -u
root=/root/autodl-tmp/sttrack_m54_template_reader_v1_20260906
cd /root/autodl-tmp/rgbd_baselines/STTrack_lachtt_v1
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python=/root/autodl-tmp/envs/sttrack/bin/python
echo $$ > "$root/controller.pid"
"$python" -m tools.collect_sttrack_m54 --root "$root" > "$root/collection.log" 2>&1
code=$?
echo "$code" > "$root/collection.exit"
if [ "$code" -eq 0 ]; then
  "$python" -m tools.train_sttrack_m54 --root "$root" > "$root/training.log" 2>&1
  code=$?
  echo "$code" > "$root/training.exit"
fi
if [ "$code" -eq 0 ]; then
  "$python" -m tools.run_sttrack_m54 --root "$root" > "$root/recursive.log" 2>&1
  code=$?
  echo "$code" > "$root/recursive.exit"
fi
if [ "$code" -eq 0 ]; then
  "$python" -m tools.run_sttrack_m54 --root "$root" --analyze > "$root/analysis.log" 2>&1
  code=$?
  echo "$code" > "$root/analysis.exit"
fi
echo "$code" > "$root/controller.exit"
exit "$code"
