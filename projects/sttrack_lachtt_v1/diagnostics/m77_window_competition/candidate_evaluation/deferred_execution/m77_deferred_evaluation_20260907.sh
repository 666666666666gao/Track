#!/bin/bash
set -u
root=/root/autodl-tmp/sttrack_m77_window_competition_20260907/candidate_evaluation/deferred_execution
cd "$root" || exit 1
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
/root/autodl-tmp/envs/sttrack/bin/python -u /root/autodl-tmp/m77_deferred_evaluation_20260907.py controller > controller.log 2>&1
code=$?
printf '%s\n' "$code" > controller.exit
exit "$code"
