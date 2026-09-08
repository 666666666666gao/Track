#!/bin/bash
set -u
root=/root/autodl-tmp/sttrack_m78_raw_competition_20260908/candidate_evaluation/full_followup
cd "$root" || exit 1
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
/root/autodl-tmp/envs/sttrack/bin/python -u /root/autodl-tmp/m78_low22_full_followup_20260908.py > controller.log 2>&1
code=$?
printf '%s\n' "$code" > controller.exit
exit "$code"
