#!/usr/bin/env bash
set -euo pipefail

root=/root/autodl-tmp/sttrack_m90_train_states_20260928
probe=/root/autodl-tmp/sttrack_native_candidate_preservation_20260926/probe
repo=/root/autodl-tmp/rgbd_baselines/STTrack_lachtt_v1
python=/root/autodl-tmp/envs/sttrack/bin/python

while [ ! -f "$probe/complete.exit" ]; do
    sleep 300
done
test "$(cat "$probe/complete.exit")" = 0
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
CUDA_VISIBLE_DEVICES=0 "$python" -u \
    /home/SUTrack_RGBD_L/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/collect_train_states.py \
    --root "$root" \
    --full152-spec /root/autodl-tmp/sttrack_full152_paired_20260925/M82/training_spec.json \
    --repository "$repo" \
    --checkpoint /root/autodl-tmp/sttrack_checkpoints/STTrack_Vot22.pth.tar \
    --shard 0 --smoke >"$root/smoke_shard0.log" 2>&1
printf '0\n' >"$root/smoke_shard0.exit"
