#!/bin/bash
set -euo pipefail
root=/root/autodl-tmp/sttrack_m55_tsg_direction_v1_20260906
arm=$1
gpu=$2
cd "$root"
test ! -e "train_${arm}_controller.pid"
printf '%s\n' "$$" > "train_${arm}_controller.pid"
set +e
CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /root/autodl-tmp/envs/sttrack/bin/python -u train.py --root "$root" --variant "$arm" --mode train > "train_${arm}.log" 2>&1
status=$?
printf '%s\n' "$status" > "train_${arm}.exit"
printf '%s\n' "$status" > "train_${arm}_controller.exit"
exit "$status"
