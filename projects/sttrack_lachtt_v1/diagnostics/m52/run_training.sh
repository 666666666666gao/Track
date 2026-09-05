#!/bin/bash
set -u
cd /root/autodl-tmp/sttrack_m52_policy_state_augmentation_v1_20260905
printf '%s\n' "$$" > training_controller.pid
run_stage() {
    stage="$1"
    shift
    "$@" > "$stage.log" 2>&1
    code=$?
    printf '%s\n' "$code" > "$stage.exit"
    if [ "$code" -ne 0 ]; then
        printf '%s\n' "$code" > pipeline.exit
        exit "$code"
    fi
}
run_stage data_audit env CUDA_VISIBLE_DEVICES= /root/autodl-tmp/envs/sttrack/bin/python /root/autodl-tmp/rgbd_baselines/STTrack_lachtt_v1/tools/train_sttrack_m52.py --root /root/autodl-tmp/sttrack_m52_policy_state_augmentation_v1_20260905 --audit
run_stage training_control env CUDA_VISIBLE_DEVICES=0 /root/autodl-tmp/envs/sttrack/bin/python /root/autodl-tmp/rgbd_baselines/STTrack_lachtt_v1/tools/train_sttrack_m52.py --root /root/autodl-tmp/sttrack_m52_policy_state_augmentation_v1_20260905 --arm control
run_stage training_mixed env CUDA_VISIBLE_DEVICES=0 /root/autodl-tmp/envs/sttrack/bin/python /root/autodl-tmp/rgbd_baselines/STTrack_lachtt_v1/tools/train_sttrack_m52.py --root /root/autodl-tmp/sttrack_m52_policy_state_augmentation_v1_20260905 --arm mixed
run_stage runtime_contract env CUDA_VISIBLE_DEVICES=0 /root/autodl-tmp/envs/sttrack/bin/python /root/autodl-tmp/sttrack_m52_policy_state_augmentation_v1_20260905/check_runtime.py
run_stage recursive_control_s0 env CUDA_VISIBLE_DEVICES=0 /root/autodl-tmp/envs/sttrack/bin/python /root/autodl-tmp/rgbd_baselines/STTrack_lachtt_v1/tools/run_sttrack_m52.py --root /root/autodl-tmp/sttrack_m52_policy_state_augmentation_v1_20260905 --arm control --shard 0
run_stage recursive_control_s1 env CUDA_VISIBLE_DEVICES=0 /root/autodl-tmp/envs/sttrack/bin/python /root/autodl-tmp/rgbd_baselines/STTrack_lachtt_v1/tools/run_sttrack_m52.py --root /root/autodl-tmp/sttrack_m52_policy_state_augmentation_v1_20260905 --arm control --shard 1
run_stage recursive_mixed_s0 env CUDA_VISIBLE_DEVICES=0 /root/autodl-tmp/envs/sttrack/bin/python /root/autodl-tmp/rgbd_baselines/STTrack_lachtt_v1/tools/run_sttrack_m52.py --root /root/autodl-tmp/sttrack_m52_policy_state_augmentation_v1_20260905 --arm mixed --shard 0
run_stage recursive_mixed_s1 env CUDA_VISIBLE_DEVICES=0 /root/autodl-tmp/envs/sttrack/bin/python /root/autodl-tmp/rgbd_baselines/STTrack_lachtt_v1/tools/run_sttrack_m52.py --root /root/autodl-tmp/sttrack_m52_policy_state_augmentation_v1_20260905 --arm mixed --shard 1
run_stage analysis env CUDA_VISIBLE_DEVICES= /root/autodl-tmp/envs/sttrack/bin/python /root/autodl-tmp/rgbd_baselines/STTrack_lachtt_v1/tools/run_sttrack_m52.py --root /root/autodl-tmp/sttrack_m52_policy_state_augmentation_v1_20260905 --analyze
printf '0\n' > pipeline.exit
