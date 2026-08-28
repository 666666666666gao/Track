#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/sutrack_vot_all127_anchor_identity_v3/run
PYTHON=/root/miniconda3/envs/mplt/bin/python
REPO=/home/SUTrack_RGBD_L
PID_FILE="$ROOT/diagnostics.pid"
LOG_FILE="$ROOT/diagnostics.nohup.log"

if [[ -f "$ROOT/full127_sequence_and_failure_diagnostics.json" ]]; then
  echo "full-127 diagnostics already exist"
  exit 0
fi

if [[ -f "$PID_FILE" ]]; then
  diagnostics_pid="$(tr -d '[:space:]' < "$PID_FILE")"
  if [[ "$diagnostics_pid" =~ ^[0-9]+$ ]] && kill -0 "$diagnostics_pid" 2>/dev/null; then
    echo "diagnostics already running: $diagnostics_pid"
    exit 0
  fi
fi

cd "$REPO"
nohup "$PYTHON" tools/finalize_vot_anchor_identity_diagnostics.py \
  --root "$ROOT" --poll-seconds 120 \
  >"$LOG_FILE" 2>&1 </dev/null &
diagnostics_pid=$!
temporary="$PID_FILE.tmp-$diagnostics_pid"
printf '%s\n' "$diagnostics_pid" > "$temporary"
mv "$temporary" "$PID_FILE"
echo "started diagnostics: $diagnostics_pid"
