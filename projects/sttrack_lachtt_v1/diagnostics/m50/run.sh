#!/bin/bash
set -u
cd /root/autodl-tmp/sttrack_m50_scale_template_v1_20260905
printf '%s\n' "$$" > controller.pid
CUDA_VISIBLE_DEVICES=0 /root/autodl-tmp/envs/sttrack/bin/python /root/autodl-tmp/rgbd_baselines/STTrack_lachtt_v1/tools/run_sttrack_m50.py --root /root/autodl-tmp/sttrack_m50_scale_template_v1_20260905 > recursive.log 2>&1
code=$?
printf '%s\n' "$code" > recursive.exit
if [ "$code" -eq 0 ]; then
/root/autodl-tmp/envs/sttrack/bin/python /root/autodl-tmp/rgbd_baselines/STTrack_lachtt_v1/tools/run_sttrack_m50.py --root /root/autodl-tmp/sttrack_m50_scale_template_v1_20260905 --analyze > analysis.log 2>&1
code=$?
printf '%s\n' "$code" > analysis.exit
fi
printf '%s\n' "$code" > controller.exit
exit "$code"
