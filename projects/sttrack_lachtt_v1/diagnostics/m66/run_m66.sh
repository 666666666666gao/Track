#!/bin/bash
set -u
root=/root/autodl-tmp/sttrack_m66_same_state_diagnostic_20260907
python=/root/autodl-tmp/envs/sttrack/bin/python
script=/root/autodl-tmp/m66_same_state_diagnostic_20260907.py
(CUDA_VISIBLE_DEVICES=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "$python" -u "$script" control > "$root/control.log" 2>&1; code=$?; echo "$code" > "$root/control.exit"; exit "$code") &
control=$!
(CUDA_VISIBLE_DEVICES=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "$python" -u "$script" null > "$root/null.log" 2>&1; code=$?; echo "$code" > "$root/null.exit"; exit "$code") &
null=$!
wait "$control"; control_code=$?
wait "$null"; null_code=$?
if [ "$control_code" -eq 0 ] && [ "$null_code" -eq 0 ]; then
  echo 0 > "$root/controller.exit"
else
  echo 1 > "$root/controller.exit"
fi
