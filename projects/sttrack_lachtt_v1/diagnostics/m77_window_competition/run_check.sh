#!/bin/bash
cd /root/autodl-tmp/sttrack_m77_window_competition_20260907 || exit 1
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
CUDA_VISIBLE_DEVICES=0 /root/autodl-tmp/envs/sttrack/bin/python -u check_causal.py > causal_check.log 2>&1
s=$?
printf "%s\n" "$s" > causal_check.exit
exit "$s"
