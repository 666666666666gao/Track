#!/bin/bash
set -u
cd /root/autodl-tmp/sttrack_default_rgbd_ope_v1_20260906
dataset="$1"
printf '%s\n' "$$" > "$dataset.controller.pid"
env CUDA_VISIBLE_DEVICES=0 /root/autodl-tmp/envs/sttrack/bin/python /root/autodl-tmp/rgbd_baselines/STTrack_lachtt_v1/tools/run_sttrack_native_ope.py --root /root/autodl-tmp/sttrack_default_rgbd_ope_v1_20260906 --mode track --dataset "$dataset" > "tracking_$dataset.log" 2>&1
code=$?
printf '%s\n' "$code" > "tracking_$dataset.exit"
if [ "$code" -eq 0 ]; then
    env CUDA_VISIBLE_DEVICES= /root/miniconda3/envs/mplt/bin/python /root/autodl-tmp/rgbd_baselines/STTrack_lachtt_v1/tools/run_sttrack_native_ope.py --root /root/autodl-tmp/sttrack_default_rgbd_ope_v1_20260906 --mode analyze --dataset "$dataset" > "analysis_$dataset.log" 2>&1
    code=$?
    printf '%s\n' "$code" > "analysis_$dataset.exit"
fi
printf '%s\n' "$code" > "$dataset.controller.exit"
exit "$code"
