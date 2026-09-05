#!/bin/bash
set -u
cd /root/autodl-tmp/sttrack_m51_relative_geometry_v1_20260905
printf '%s\n' "$$" > controller.pid
export CUDA_VISIBLE_DEVICES=0
run_stage() {
    stage="$1"
    shift
    "$@" > "$stage.log" 2>&1
    code=$?
    printf '%s\n' "$code" > "$stage.exit"
    if [ "$code" -ne 0 ]; then
        printf '%s\n' "$code" > controller.exit
        exit "$code"
    fi
}
run_stage training /root/autodl-tmp/envs/sttrack/bin/python /root/autodl-tmp/rgbd_baselines/STTrack_lachtt_v1/tools/train_sttrack_m51.py --root /root/autodl-tmp/sttrack_m51_relative_geometry_v1_20260905
run_stage runtime_contract /root/autodl-tmp/envs/sttrack/bin/python /root/autodl-tmp/sttrack_m51_relative_geometry_v1_20260905/check_runtime.py
run_stage recursive_s0 /root/autodl-tmp/envs/sttrack/bin/python /root/autodl-tmp/rgbd_baselines/STTrack_lachtt_v1/tools/run_sttrack_m51.py --root /root/autodl-tmp/sttrack_m51_relative_geometry_v1_20260905 --shard 0
run_stage recursive_s1 /root/autodl-tmp/envs/sttrack/bin/python /root/autodl-tmp/rgbd_baselines/STTrack_lachtt_v1/tools/run_sttrack_m51.py --root /root/autodl-tmp/sttrack_m51_relative_geometry_v1_20260905 --shard 1
run_stage analysis /root/autodl-tmp/envs/sttrack/bin/python /root/autodl-tmp/rgbd_baselines/STTrack_lachtt_v1/tools/run_sttrack_m51.py --root /root/autodl-tmp/sttrack_m51_relative_geometry_v1_20260905 --analyze
printf '0\n' > controller.exit
