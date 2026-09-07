#!/bin/bash
set -u
cd /root/autodl-tmp/sttrack_m73_paired_lexical_replication_20260907/completion_queue || exit 1
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
/root/autodl-tmp/envs/sttrack/bin/python -u /root/autodl-tmp/m73_completion_queue_20260907.py run > controller.log 2>&1
status=$?; printf '%s\n' "$status" > controller.exit
exit "$status"
