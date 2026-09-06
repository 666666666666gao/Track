#!/usr/bin/env bash
set -u
root=/root/autodl-tmp/sttrack_m55_tsg_direction_v1_20260906
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$root"
echo $$ > "$root/contract_v2/controller.pid"
status=0
for variant in control clone; do
  /root/autodl-tmp/envs/sttrack/bin/python "$root/check_training_v2.py" --root "$root" --variant "$variant" > "$root/contract_v2/$variant.log" 2>&1
  code=$?
  echo "$code" > "$root/contract_v2/$variant.exit"
  if [ "$code" -ne 0 ]; then status=1; fi
done
echo "$status" > "$root/contract_v2/controller.exit"
exit "$status"
