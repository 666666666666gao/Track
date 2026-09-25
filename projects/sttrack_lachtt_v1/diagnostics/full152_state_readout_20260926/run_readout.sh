#!/usr/bin/env bash
set -u

root=/root/autodl-tmp/sttrack_full152_state_readout_20260926
model_python=/root/autodl-tmp/envs/sttrack/bin/python
metric_python=/root/miniconda3/envs/mplt/bin/python
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$root" || exit 1

CUDA_VISIBLE_DEVICES=1 "$model_python" -u readout.py > readout.log 2>&1
status=$?
printf '%s\n' "$status" > readout.exit
if [ "$status" -ne 0 ]; then exit "$status"; fi

"$metric_python" -u analyze.py > analyze.log 2>&1
status=$?
printf '%s\n' "$status" > analyze.exit
exit "$status"
