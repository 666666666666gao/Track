#!/usr/bin/env bash
set -u
cd /root/autodl-tmp/sttrack_m61_fixed_state_language_20260907
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
CUDA_VISIBLE_DEVICES=0 /root/autodl-tmp/envs/sttrack/bin/python /root/autodl-tmp/m61_fixed_state_language_20260907.py prefix --reference original > prefix_original.log 2>&1 &
m61_p0=$!
CUDA_VISIBLE_DEVICES=1 /root/autodl-tmp/envs/sttrack/bin/python /root/autodl-tmp/m61_fixed_state_language_20260907.py prefix --reference category > prefix_category.log 2>&1 &
m61_p1=$!
wait "$m61_p0"
m61_s0=$?
printf '%s\n' "$m61_s0" > prefix_original.exit
wait "$m61_p1"
m61_s1=$?
printf '%s\n' "$m61_s1" > prefix_category.exit
if [ "$m61_s0" -ne 0 ] || [ "$m61_s1" -ne 0 ]; then
  printf '1\n' > prefix_pair.exit
  exit 1
fi
printf '0\n' > prefix_pair.exit
