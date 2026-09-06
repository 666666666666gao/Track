#!/bin/bash
set -e
echo $$ > /root/autodl-tmp/sttrack_m57_initial_binding_v1_20260906/training.pid
trap 'status=$?; echo "$status" > /root/autodl-tmp/sttrack_m57_initial_binding_v1_20260906/training.exit' EXIT
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
cd /root/autodl-tmp/sttrack_m57_initial_binding_v1_20260906/code
/root/autodl-tmp/envs/sttrack/bin/python /root/autodl-tmp/sttrack_m57_initial_binding_v1_20260906/train.py --root /root/autodl-tmp/sttrack_m57_initial_binding_v1_20260906 > /root/autodl-tmp/sttrack_m57_initial_binding_v1_20260906/training.log 2>&1
