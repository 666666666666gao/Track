#!/usr/bin/env bash
set -uo pipefail
SRC=/home/SUTrack_RGBD_L/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928
OUT=/root/autodl-tmp/sttrack_m98_train_contexts_20260928
PY=/root/autodl-tmp/envs/sttrack/bin/python
cd "$SRC" || exit 1
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
COMMON=(--cache /root/autodl-tmp/sttrack_m90_train_states_20260928
        --origins /root/autodl-tmp/sttrack_m95_initial_origins_20260928
        --output "$OUT"
        --full152-spec /root/autodl-tmp/sttrack_full152_paired_20260925/M82/training_spec.json
        --repository /root/autodl-tmp/rgbd_baselines/STTrack_lachtt_v1
        --checkpoint /root/autodl-tmp/sttrack_checkpoints/STTrack_Vot22.pth.tar)
CUDA_VISIBLE_DEVICES=0 "$PY" -u collect_train_contexts.py "${COMMON[@]}" --shard 0 > "$OUT/shard0.log" 2>&1 &
PID0=$!
CUDA_VISIBLE_DEVICES=1 "$PY" -u collect_train_contexts.py "${COMMON[@]}" --shard 1 > "$OUT/shard1.log" 2>&1 &
PID1=$!
printf '%s\n' "$PID0" > "$OUT/shard0.pid"
printf '%s\n' "$PID1" > "$OUT/shard1.pid"
wait "$PID0"; EXIT0=$?
wait "$PID1"; EXIT1=$?
printf '%s\n' "$EXIT0" > "$OUT/shard0.exit"
printf '%s\n' "$EXIT1" > "$OUT/shard1.exit"
if [ "$EXIT0" -ne 0 ] || [ "$EXIT1" -ne 0 ]; then
  printf '1\n' > "$OUT/queue.exit"
  exit 1
fi
CUDA_VISIBLE_DEVICES='' "$PY" -u audit_train_contexts.py --cache /root/autodl-tmp/sttrack_m90_train_states_20260928 --contexts "$OUT" > "$OUT/input_audit.log" 2>&1
EXIT_AUDIT=$?
printf '%s\n' "$EXIT_AUDIT" > "$OUT/queue.exit"
exit "$EXIT_AUDIT"
