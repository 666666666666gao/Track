#!/bin/bash
set -e
echo $$ > /root/autodl-tmp/sttrack_m57_initial_binding_v1_20260906/recursive_queue0.pid
trap 'status=$?; echo "$status" > /root/autodl-tmp/sttrack_m57_initial_binding_v1_20260906/recursive_queue0.exit' EXIT
export PYTHONUNBUFFERED=1
cd /root/autodl-tmp/sttrack_m57_initial_binding_v1_20260906
/root/autodl-tmp/envs/sttrack/bin/python /root/autodl-tmp/sttrack_m57_initial_binding_v1_20260906/queue_recursive.py --gpu 0 > /root/autodl-tmp/sttrack_m57_initial_binding_v1_20260906/recursive_queue0.log 2>&1
