#!/bin/bash
set -e
echo $$ > /root/autodl-tmp/sttrack_m57_initial_binding_v1_20260906/analysis_scheduler.pid
trap 'status=$?; echo "$status" > /root/autodl-tmp/sttrack_m57_initial_binding_v1_20260906/analysis_scheduler.exit' EXIT
export CUDA_VISIBLE_DEVICES=""
export PYTHONUNBUFFERED=1
cd /root/autodl-tmp/sttrack_m57_initial_binding_v1_20260906
/root/autodl-tmp/envs/sttrack/bin/python /root/autodl-tmp/sttrack_m57_initial_binding_v1_20260906/finish_recursive.py > /root/autodl-tmp/sttrack_m57_initial_binding_v1_20260906/analysis_scheduler.log 2>&1
