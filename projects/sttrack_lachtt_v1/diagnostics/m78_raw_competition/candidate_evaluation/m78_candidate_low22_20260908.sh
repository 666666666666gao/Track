#!/bin/bash
set -u
root=/root/autodl-tmp/sttrack_m78_raw_competition_20260908/candidate_evaluation
model=/root/autodl-tmp/envs/sttrack/bin/python
metric=/root/miniconda3/envs/mplt/bin/python
script=/root/autodl-tmp/m78_vot_low22_20260908.py
cd "$root" || exit 1
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
CUDA_VISIBLE_DEVICES='' "$model" -u "$script" prepare > low22_preparation.log 2>&1
code=$?
printf '%s\n' "$code" > low22_preparation.exit
if [ "$code" -ne 0 ]; then printf '%s\n' "$code" > low22_controller.exit; exit "$code"; fi
CUDA_VISIBLE_DEVICES='' "$model" -u "$script" check > low22_binding.log 2>&1
code=$?
printf '%s\n' "$code" > low22_binding.exit
if [ "$code" -eq 0 ]; then
    CUDA_VISIBLE_DEVICES='' "$metric" -u run_vot_failure_family_shards.py --root "$root/low22_run" --poll-seconds 240 > low22_tracking.log 2>&1
    code=$?
    printf '%s\n' "$code" > low22_tracking.exit
fi
if [ "$code" -eq 0 ]; then
    CUDA_VISIBLE_DEVICES='' "$metric" -u "$script" analyze > low22_analysis.log 2>&1
    code=$?
    printf '%s\n' "$code" > low22_analysis.exit
fi
printf '%s\n' "$code" > low22_controller.exit
exit "$code"
