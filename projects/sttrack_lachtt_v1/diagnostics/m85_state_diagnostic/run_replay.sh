#!/bin/bash
set -u
cd /root/autodl-tmp/sttrack_m85_state_diagnostic_20260921 || exit 1
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0
/root/autodl-tmp/envs/sttrack/bin/python -u same_state.py --preflight > preflight.log 2>&1
status=$?
printf '%s\n' "$status" > preflight.exit
if [ "$status" -ne 0 ]; then printf '1\n' > controller.exit; exit 1; fi
/root/autodl-tmp/envs/sttrack/bin/python -u same_state.py > replay.log 2>&1
status=$?
printf '%s\n' "$status" > replay.exit
printf '%s\n' "$status" > controller.exit
exit "$status"
