#!/usr/bin/env bash
set -u
cd /root/autodl-tmp/sttrack_m62_learned_entry_parity_20260907
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
CUDA_VISIBLE_DEVICES=0 /root/autodl-tmp/envs/sttrack/bin/python /root/autodl-tmp/sttrack_m58_evaluation_preparation_20260906/run_semantic_ope.py --plan ope_plan.json --mode track > ope.log 2>&1 &
m62_p0=$!
CUDA_VISIBLE_DEVICES='' /root/miniconda3/envs/mplt/bin/python /root/autodl-tmp/m62_learned_entry_parity_20260907.py client > trax.log 2>&1 &
m62_p1=$!
wait "$m62_p0"
m62_s0=$?
printf '%s\n' "$m62_s0" > ope.exit
wait "$m62_p1"
m62_s1=$?
printf '%s\n' "$m62_s1" > trax.exit
if [ "$m62_s0" -ne 0 ] || [ "$m62_s1" -ne 0 ]; then
  printf '1\n' > job.exit
  exit 1
fi
CUDA_VISIBLE_DEVICES='' /root/autodl-tmp/envs/sttrack/bin/python /root/autodl-tmp/m62_learned_entry_parity_20260907.py verify > verification.log 2>&1
m62_verify_exit=$?
printf '%s\n' "$m62_verify_exit" > verification.exit
printf '%s\n' "$m62_verify_exit" > job.exit
exit "$m62_verify_exit"
