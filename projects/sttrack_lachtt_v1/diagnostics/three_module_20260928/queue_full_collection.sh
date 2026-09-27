#!/usr/bin/env bash
set -euo pipefail

root=/root/autodl-tmp/sttrack_m90_train_states_20260928
repo=/root/autodl-tmp/rgbd_baselines/STTrack_lachtt_v1
source=/home/SUTrack_RGBD_L/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928
python=/root/autodl-tmp/envs/sttrack/bin/python
trap 'status=$?; printf "%s\n" "$status" >"$root/full_collection.exit"' EXIT

while [ ! -f "$root/smoke_shard0.exit" ]; do
    sleep 300
done
test "$(cat "$root/smoke_shard0.exit")" = 0
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
"$python" -u "$source/verify_smoke.py" --root "$root" >"$root/smoke_verification.log" 2>&1

CUDA_VISIBLE_DEVICES=0 "$python" -u "$source/collect_train_states.py" \
    --root "$root" \
    --full152-spec /root/autodl-tmp/sttrack_full152_paired_20260925/M82/training_spec.json \
    --repository "$repo" \
    --checkpoint /root/autodl-tmp/sttrack_checkpoints/STTrack_Vot22.pth.tar \
    --shard 0 >"$root/collect_shard0.log" 2>&1 &
pid0=$!
CUDA_VISIBLE_DEVICES=1 "$python" -u "$source/collect_train_states.py" \
    --root "$root" \
    --full152-spec /root/autodl-tmp/sttrack_full152_paired_20260925/M82/training_spec.json \
    --repository "$repo" \
    --checkpoint /root/autodl-tmp/sttrack_checkpoints/STTrack_Vot22.pth.tar \
    --shard 1 >"$root/collect_shard1.log" 2>&1 &
pid1=$!
set +e
wait "$pid0"
status0=$?
wait "$pid1"
status1=$?
set -e
printf '%s\n' "$status0" >"$root/collect_shard0.exit"
printf '%s\n' "$status1" >"$root/collect_shard1.exit"
test "$status0" = 0
test "$status1" = 0
"$python" -u "$source/analyze_train_states.py" --root "$root" >"$root/candidate_coverage.log" 2>&1
