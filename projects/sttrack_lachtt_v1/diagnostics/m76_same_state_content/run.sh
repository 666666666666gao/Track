#!/bin/bash
set -u
cd /root/autodl-tmp/sttrack_m76_same_state_content_20260907 || exit 1
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
CUDA_VISIBLE_DEVICES=0 /root/autodl-tmp/envs/sttrack/bin/python -u /root/autodl-tmp/m76_same_state_content_20260907.py run > replay.log 2>&1
status=$?; printf '%s\n' "$status" > replay.exit
if [ "$status" -ne 0 ]; then printf '%s\n' "$status" > controller.exit; exit "$status"; fi
CUDA_VISIBLE_DEVICES='' /root/autodl-tmp/envs/sttrack/bin/python -u /root/autodl-tmp/m76_same_state_content_20260907.py analyze > analysis.log 2>&1
status=$?; printf '%s\n' "$status" > analysis.exit; printf '%s\n' "$status" > controller.exit
exit "$status"
