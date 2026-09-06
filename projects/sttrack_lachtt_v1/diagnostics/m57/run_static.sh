#!/bin/bash
set -e
echo $$ > /root/autodl-tmp/sttrack_m57_initial_binding_v1_20260906/static.pid
trap 'status=$?; echo "$status" > /root/autodl-tmp/sttrack_m57_initial_binding_v1_20260906/static.exit' EXIT
export CUDA_VISIBLE_DEVICES=""
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
cd /root/autodl-tmp/sttrack_m57_initial_binding_v1_20260906
/root/autodl-tmp/envs/sttrack/bin/python /root/autodl-tmp/sttrack_m57_initial_binding_v1_20260906/analyze_static.py > /root/autodl-tmp/sttrack_m57_initial_binding_v1_20260906/static.log 2>&1
