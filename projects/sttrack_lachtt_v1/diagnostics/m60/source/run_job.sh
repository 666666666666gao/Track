#!/usr/bin/env bash
set +e
cd /root/autodl-tmp/sttrack_m60_category_isolation_20260906
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
/root/autodl-tmp/envs/sttrack/bin/python /root/autodl-tmp/m60_category_isolation_20260906.py parity > parity.log 2>&1
run_status=$?
printf '%s\n' "$run_status" > parity.exit
if [ "$run_status" -ne 0 ]; then
    printf '%s\n' "$run_status" > job.exit
    exit "$run_status"
fi
/root/autodl-tmp/envs/sttrack/bin/python /root/autodl-tmp/m60_category_isolation_20260906.py run > tracking.log 2>&1
run_status=$?
printf '%s\n' "$run_status" > tracking.exit
if [ "$run_status" -ne 0 ]; then
    printf '%s\n' "$run_status" > job.exit
    exit "$run_status"
fi
export CUDA_VISIBLE_DEVICES=''
/root/autodl-tmp/envs/sttrack/bin/python /root/autodl-tmp/m60_category_isolation_20260906.py analyze > analysis.log 2>&1
run_status=$?
printf '%s\n' "$run_status" > analysis.exit
printf '%s\n' "$run_status" > job.exit
exit "$run_status"
