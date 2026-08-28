#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/sutrack_transaction_low22_v1/run
TRACE_ROOT="$ROOT/transaction_traces"
BASELINE_MANIFEST=/root/autodl-tmp/sutrack_vot_low22_anchor_identity_v1/run/shard_manifest.json
BASELINE_REPORT=/root/autodl-tmp/sutrack_vot_low22_anchor_identity_v1/LOW22_REPORT.json
FULL_RESULT=/root/autodl-tmp/sutrack_vot_all127_anchor_identity_v3/run/full_result.json
PYTHON=/root/miniconda3/envs/mplt/bin/python
REPO=/home/SUTrack_RGBD_L
SMOKE=/root/autodl-tmp/sutrack_transaction_low22_v1/structural_smoke.json
GPU_SMOKE=/root/autodl-tmp/sutrack_transaction_low22_v1/gpu_inference_smoke.json
GPU_SMOKE_TRACE_ROOT=/root/autodl-tmp/sutrack_transaction_low22_v1/gpu_smoke_traces
PID_FILE="$ROOT/controller.pid"
LOG_FILE="$ROOT/controller.nohup.log"
FINALIZER_PID_FILE="$ROOT/finalizer.pid"
FINALIZER_LOG_FILE="$ROOT/finalizer.nohup.log"
DIAGNOSTICS_PID_FILE="$ROOT/diagnostics.pid"
DIAGNOSTICS_LOG_FILE="$ROOT/diagnostics.nohup.log"

if [[ ! -f "$FULL_RESULT" ]]; then
  echo "refusing transaction low22 launch while anchor-text full127 is active"
  exit 3
fi
gate_complete=false
if [[ -f "$ROOT/low22_gate_result.json" ]]; then
  gate_complete=true
fi
controller_running=false
if [[ -f "$PID_FILE" ]]; then
  controller_pid="$(tr -d '[:space:]' < "$PID_FILE")"
  if [[ "$controller_pid" =~ ^[0-9]+$ ]] && kill -0 "$controller_pid" 2>/dev/null; then
    controller_running=true
  fi
fi

cd "$REPO"
if [[ ! -f "$ROOT/shard_manifest.json" ]]; then
  SUTRACK_TRANSACTION_TRACE_ROOT="$TRACE_ROOT" \
    "$PYTHON" tools/smoke_sutrack_transaction_integration.py \
    --output-json "$SMOKE"
  CUDA_VISIBLE_DEVICES="${SUTRACK_TRANSACTION_SMOKE_GPU:-0}" \
    "$PYTHON" tools/smoke_sutrack_transaction_gpu.py \
    --output-json "$GPU_SMOKE" \
    --trace-root "$GPU_SMOKE_TRACE_ROOT"
  "$PYTHON" tools/prepare_vot_transaction_low22.py \
    --output-root "$ROOT" \
    --baseline-manifest "$BASELINE_MANIFEST" \
    --baseline-report "$BASELINE_REPORT" \
    --trace-root "$TRACE_ROOT"
fi
mkdir -p "$TRACE_ROOT"
export SUTRACK_TRANSACTION_TRACE_ROOT="$TRACE_ROOT"
diagnostics_running=false
if [[ -f "$DIAGNOSTICS_PID_FILE" ]]; then
  diagnostics_pid="$(tr -d '[:space:]' < "$DIAGNOSTICS_PID_FILE")"
  if [[ "$diagnostics_pid" =~ ^[0-9]+$ ]] && kill -0 "$diagnostics_pid" 2>/dev/null; then
    diagnostics_running=true
  fi
fi
if [[ ! -f "$ROOT/low22_transaction_diagnostics.json" ]] && \
   [[ "$diagnostics_running" == false ]]; then
  nohup "$PYTHON" tools/finalize_vot_transaction_low22_diagnostics.py \
    --root "$ROOT" --poll-seconds 120 \
    >"$DIAGNOSTICS_LOG_FILE" 2>&1 </dev/null &
  diagnostics_pid=$!
  temporary="$DIAGNOSTICS_PID_FILE.tmp-$diagnostics_pid"
  printf '%s\n' "$diagnostics_pid" > "$temporary"
  mv "$temporary" "$DIAGNOSTICS_PID_FILE"
fi
if [[ "$gate_complete" == true ]]; then
  echo "transaction low22 gate complete; diagnostics is complete or running"
  exit 0
fi
if [[ ! -f "$ROOT/low22_gate_result.json" ]]; then
  finalizer_running=false
  if [[ -f "$FINALIZER_PID_FILE" ]]; then
    finalizer_pid="$(tr -d '[:space:]' < "$FINALIZER_PID_FILE")"
    if [[ "$finalizer_pid" =~ ^[0-9]+$ ]] && kill -0 "$finalizer_pid" 2>/dev/null; then
      finalizer_running=true
    fi
  fi
  if [[ "$finalizer_running" == false ]]; then
    nohup "$PYTHON" tools/finalize_vot_transaction_low22.py \
      --root "$ROOT" --poll-seconds 60 \
      >"$FINALIZER_LOG_FILE" 2>&1 </dev/null &
    finalizer_pid=$!
    temporary="$FINALIZER_PID_FILE.tmp-$finalizer_pid"
    printf '%s\n' "$finalizer_pid" > "$temporary"
    mv "$temporary" "$FINALIZER_PID_FILE"
  fi
fi
if [[ -f "$ROOT/merge_result.json" ]]; then
  echo "transaction low22 merged; finalizer is running or was restarted"
  exit 0
fi
if [[ "$controller_running" == true ]]; then
  echo "transaction low22 controller already running: $controller_pid"
  exit 0
fi
nohup "$PYTHON" tools/run_vot_failure_family_shards.py \
  --root "$ROOT" --poll-seconds 30 \
  >"$LOG_FILE" 2>&1 </dev/null &
controller_pid=$!
temporary="$PID_FILE.tmp-$controller_pid"
printf '%s\n' "$controller_pid" > "$temporary"
mv "$temporary" "$PID_FILE"
echo "started transaction low22 controller: $controller_pid"
