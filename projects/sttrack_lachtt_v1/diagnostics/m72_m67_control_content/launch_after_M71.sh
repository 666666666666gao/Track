#!/bin/bash
set -u
cd /root/autodl-tmp/sttrack_m72_m67_control_content_20260907 || exit 1
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
/root/autodl-tmp/envs/sttrack/bin/python -u /root/autodl-tmp/m72_m67_control_content_20260907.py wait_parent > wait.log 2>&1
status=$?; printf '%s\n' "$status" > wait.exit
if [ "$status" -eq 0 ]; then bash run_controls.sh > controller.log 2>&1; status=$?; fi
printf '%s\n' "$status" > after_parent.exit
exit "$status"
