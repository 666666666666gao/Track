#!/usr/bin/env bash
set -u
root=/root/autodl-tmp/sttrack_m54_template_reader_v1_20260906
cd /root/autodl-tmp/rgbd_baselines/STTrack_lachtt_v1
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
echo $$ > "$root/contract_controller.pid"
/root/autodl-tmp/envs/sttrack/bin/python -m tools.collect_sttrack_m54 --root "$root" --contract > "$root/contract.log" 2>&1
code=$?
echo "$code" > "$root/contract.exit"
if [ "$code" -eq 0 ]; then
  /root/autodl-tmp/envs/sttrack/bin/python -m tools.check_sttrack_m54 --root "$root" > "$root/runtime_contract.log" 2>&1
  code=$?
  echo "$code" > "$root/runtime_contract.exit"
fi
echo "$code" > "$root/contract_controller.exit"
exit "$code"
