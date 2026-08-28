#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/sutrack_vot_all127_anchor_identity_v3/run
PYTHON=/root/miniconda3/envs/mplt/bin/python
REPO=/home/SUTrack_RGBD_L
PID_FILE="$ROOT/controller.pid"
LOG_FILE="$ROOT/controller.nohup.log"

if [[ -f "$ROOT/merge_result.json" ]]; then
  echo "full-127 result already merged"
  exit 0
fi

if [[ -f "$PID_FILE" ]]; then
  controller_pid="$(tr -d '[:space:]' < "$PID_FILE")"
  if [[ "$controller_pid" =~ ^[0-9]+$ ]] && kill -0 "$controller_pid" 2>/dev/null; then
    echo "controller already running: $controller_pid"
    exit 0
  fi
fi

cd "$REPO"
nohup "$PYTHON" tools/run_vot_failure_family_shards.py \
  --root "$ROOT" --poll-seconds 30 \
  >"$LOG_FILE" 2>&1 </dev/null &
controller_pid=$!
temporary="$PID_FILE.tmp-$controller_pid"
printf '%s\n' "$controller_pid" > "$temporary"
mv "$temporary" "$PID_FILE"
echo "started controller: $controller_pid"
