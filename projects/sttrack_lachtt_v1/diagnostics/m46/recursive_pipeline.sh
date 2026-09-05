#!/bin/bash
set -u
cd /root/autodl-tmp/rgbd_baselines/STTrack_lachtt_v1
echo $$ > /root/autodl-tmp/sttrack_m46_initial_frame_v1_20260905/recursive_controller.pid
CUDA_VISIBLE_DEVICES=0 /root/autodl-tmp/envs/sttrack/bin/python tools/run_sttrack_m46.py --root /root/autodl-tmp/sttrack_m46_initial_frame_v1_20260905 --shard 0 > /root/autodl-tmp/sttrack_m46_initial_frame_v1_20260905/recursive_s0.log 2>&1 &
recursive0=$!
echo $recursive0 > /root/autodl-tmp/sttrack_m46_initial_frame_v1_20260905/recursive_s0.pid
CUDA_VISIBLE_DEVICES=1 /root/autodl-tmp/envs/sttrack/bin/python tools/run_sttrack_m46.py --root /root/autodl-tmp/sttrack_m46_initial_frame_v1_20260905 --shard 1 > /root/autodl-tmp/sttrack_m46_initial_frame_v1_20260905/recursive_s1.log 2>&1 &
recursive1=$!
echo $recursive1 > /root/autodl-tmp/sttrack_m46_initial_frame_v1_20260905/recursive_s1.pid
wait $recursive0
status0=$?
echo $status0 > /root/autodl-tmp/sttrack_m46_initial_frame_v1_20260905/recursive_s0.exit
wait $recursive1
status1=$?
echo $status1 > /root/autodl-tmp/sttrack_m46_initial_frame_v1_20260905/recursive_s1.exit
if [ $status0 -ne 0 ] || [ $status1 -ne 0 ]; then echo 1 > /root/autodl-tmp/sttrack_m46_initial_frame_v1_20260905/recursive_controller.exit; exit 1; fi
CUDA_VISIBLE_DEVICES='' /root/autodl-tmp/envs/sttrack/bin/python tools/run_sttrack_m46.py --root /root/autodl-tmp/sttrack_m46_initial_frame_v1_20260905 --analyze > /root/autodl-tmp/sttrack_m46_initial_frame_v1_20260905/analysis.log 2>&1
analysis_status=$?
echo $analysis_status > /root/autodl-tmp/sttrack_m46_initial_frame_v1_20260905/recursive_controller.exit
exit $analysis_status
