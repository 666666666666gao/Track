#!/bin/bash
CUDA_VISIBLE_DEVICES=0 /root/autodl-tmp/envs/sttrack/bin/python -u /root/autodl-tmp/sttrack_m57_initial_binding_v1_20260906/check_runtime_inputs.py --spec /root/autodl-tmp/sttrack_m57_initial_binding_v1_20260906/runtime_contract_spec.json > /root/autodl-tmp/sttrack_m57_initial_binding_v1_20260906/runtime_contract.log 2>&1 &
child=$!
printf "%s\n" "$child" > /root/autodl-tmp/sttrack_m57_initial_binding_v1_20260906/runtime_contract.pid
wait "$child"
status=$?
printf "%s\n" "$status" > /root/autodl-tmp/sttrack_m57_initial_binding_v1_20260906/runtime_contract.exit
exit "$status"
