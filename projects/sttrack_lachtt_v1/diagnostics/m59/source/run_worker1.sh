#!/usr/bin/env bash
set +e
cd /root/autodl-tmp/sttrack_m59_fixed_head_content_diagnostic_20260906
export CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
/root/autodl-tmp/envs/sttrack/bin/python run_controls.py --worker 1 > worker1.log 2>&1
run_status=$?
printf '%s\n' "$run_status" > worker1.exit
exit "$run_status"
