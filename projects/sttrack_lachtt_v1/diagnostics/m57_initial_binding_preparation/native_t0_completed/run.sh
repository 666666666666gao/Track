#!/bin/bash
cd /root/autodl-tmp/sttrack_m55_tsg_direction_v2_20260906/code/control
CUDA_VISIBLE_DEVICES=1 /root/autodl-tmp/envs/sttrack/bin/python -u /root/autodl-tmp/sttrack_m57_t0_contract_preparation_20260906/verify_t0_extraction.py --spec /root/autodl-tmp/sttrack_m57_native_t0_contract_20260906/spec.json > /root/autodl-tmp/sttrack_m57_native_t0_contract_20260906/run.log 2>&1 &
child=$!
printf "%s\n" "$child" > /root/autodl-tmp/sttrack_m57_native_t0_contract_20260906/run.pid
wait "$child"
status=$?
printf "%s\n" "$status" > /root/autodl-tmp/sttrack_m57_native_t0_contract_20260906/run.exit
exit "$status"
