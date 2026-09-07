#!/bin/bash
set -u
root=/root/autodl-tmp/sttrack_post_m67_diagnostic_queue_20260907
cd "$root" || exit 1
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
/root/autodl-tmp/envs/sttrack/bin/python -u /root/autodl-tmp/post_m67_diagnostic_queue_20260907.py run > controller.log 2>&1
status=$?
printf '%s\n' "$status" > controller.exit
exit "$status"
