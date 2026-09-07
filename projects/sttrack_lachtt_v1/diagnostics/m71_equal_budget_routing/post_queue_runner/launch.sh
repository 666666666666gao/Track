#!/bin/bash
set -u
cd /root/autodl-tmp/sttrack_m71_equal_budget_routing_20260907/post_queue_runner || exit 1
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
/root/autodl-tmp/envs/sttrack/bin/python -u /root/autodl-tmp/m71_after_diagnostics_20260907.py run > controller.log 2>&1
status=$?
printf '%s\n' "$status" > controller.exit
exit "$status"
