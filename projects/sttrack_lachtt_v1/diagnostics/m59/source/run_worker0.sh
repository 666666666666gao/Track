#!/usr/bin/env bash
set +e
cd /root/autodl-tmp/sttrack_m59_fixed_head_content_diagnostic_20260906
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
/root/autodl-tmp/envs/sttrack/bin/python run_controls.py --worker 0 > worker0.log 2>&1
run_status=$?
printf '%s\n' "$run_status" > worker0.exit
if [ "$run_status" -eq 0 ]; then
    export CUDA_VISIBLE_DEVICES=''
    /root/autodl-tmp/envs/sttrack/bin/python finalize.py > finalization.log 2>&1
    run_status=$?
    printf '%s\n' "$run_status" > finalization.exit
fi
exit "$run_status"
