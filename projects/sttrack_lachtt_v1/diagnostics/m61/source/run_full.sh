#!/usr/bin/env bash
set -u
cd /root/autodl-tmp/sttrack_m61_fixed_state_language_20260907
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
CUDA_VISIBLE_DEVICES=0 /root/autodl-tmp/envs/sttrack/bin/python /root/autodl-tmp/m61_fixed_state_language_20260907.py collect --reference original > collect_original.log 2>&1 &
m61_p0=$!
CUDA_VISIBLE_DEVICES=1 /root/autodl-tmp/envs/sttrack/bin/python /root/autodl-tmp/m61_fixed_state_language_20260907.py collect --reference category > collect_category.log 2>&1 &
m61_p1=$!
wait "$m61_p0"
m61_s0=$?
printf '%s\n' "$m61_s0" > collect_original.exit
wait "$m61_p1"
m61_s1=$?
printf '%s\n' "$m61_s1" > collect_category.exit
if [ "$m61_s0" -ne 0 ] || [ "$m61_s1" -ne 0 ]; then
  printf '1\n' > job.exit
  exit 1
fi
CUDA_VISIBLE_DEVICES='' /root/autodl-tmp/envs/sttrack/bin/python /root/autodl-tmp/analyze_m61_fixed_state_20260907.py > analysis.log 2>&1
m61_analysis_exit=$?
printf '%s\n' "$m61_analysis_exit" > analysis.exit
printf '%s\n' "$m61_analysis_exit" > job.exit
exit "$m61_analysis_exit"
