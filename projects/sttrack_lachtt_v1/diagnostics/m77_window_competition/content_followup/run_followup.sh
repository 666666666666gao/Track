#!/bin/bash
cd /root/autodl-tmp/sttrack_m77_window_competition_20260907/content_followup || exit 1
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
CUDA_VISIBLE_DEVICES="" /root/autodl-tmp/envs/sttrack/bin/python -u /root/autodl-tmp/m77_content_followup_20260907.py controller > controller.log 2>&1
status=$?
printf "%s\n" "$status" > controller.exit
exit "$status"
