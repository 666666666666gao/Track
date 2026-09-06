#!/bin/bash
set -uo pipefail
root=/root/autodl-tmp/sttrack_m55_tsg_direction_v2_20260906
cd "$root"
printf '%s\n' "$$" > sampler_contract_controller.pid
for variant in control clone; do
  CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /root/autodl-tmp/envs/sttrack/bin/python -u train.py --root "$root" --variant "$variant" --mode contract > "sampler_contract_${variant}.log" 2>&1
  status=$?
  printf '%s\n' "$status" > "sampler_contract_${variant}.exit"
  if [ "$status" -ne 0 ]; then
    printf '%s\n' "$status" > sampler_contract_controller.exit
    exit "$status"
  fi
done
printf '0\n' > sampler_contract_controller.exit
