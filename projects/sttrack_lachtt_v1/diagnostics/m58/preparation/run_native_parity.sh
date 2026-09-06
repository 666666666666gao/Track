#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
/root/autodl-tmp/envs/sttrack/bin/python -u /root/autodl-tmp/sttrack_m58_semantic_spatial_v1_20260906/check_native_parity.py > /root/autodl-tmp/sttrack_m58_semantic_spatial_v1_20260906/native_parity.log 2>&1
task_status=$?
printf '%s\n' "$task_status" > /root/autodl-tmp/sttrack_m58_semantic_spatial_v1_20260906/native_parity.exit
exit "$task_status"
