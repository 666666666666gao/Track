#!/bin/bash
set -u
cd /root/autodl-tmp/sttrack_m53_template_read_capacity_v1_20260905
printf '%s\n' "$$" > analysis_controller.pid
sleep "$1"
while [ ! -f controller.exit ]; do
    sleep 240
done
env CUDA_VISIBLE_DEVICES= /root/autodl-tmp/envs/sttrack/bin/python /root/autodl-tmp/rgbd_baselines/STTrack_lachtt_v1/tools/analyze_sttrack_m53.py --root /root/autodl-tmp/sttrack_m53_template_read_capacity_v1_20260905 > analysis.log 2>&1
code=$?
printf '%s\n' "$code" > analysis.exit
exit "$code"
