#!/bin/bash
set -u
cd /root/autodl-tmp/sttrack_m52_policy_state_augmentation_v1_20260905
printf '%s\n' "$$" > controller.pid
CUDA_VISIBLE_DEVICES=0 /root/autodl-tmp/envs/sttrack/bin/python /root/autodl-tmp/rgbd_baselines/STTrack_lachtt_v1/tools/collect_sttrack_m52.py --root /root/autodl-tmp/sttrack_m52_policy_state_augmentation_v1_20260905 > collection.log 2>&1
code=$?
printf '%s\n' "$code" > collection.exit
printf '%s\n' "$code" > controller.exit
exit "$code"
