#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/sutrack_vot_all127_anchor_identity_v3/run
PYTHON=/root/miniconda3/envs/mplt/bin/python
REPO=/home/SUTrack_RGBD_L
PID_FILE="$ROOT/finalizer.pid"
LOG_FILE="$ROOT/finalizer.nohup.log"

if [[ -f "$ROOT/full_result.json" ]]; then
  echo "full-127 terminal result already exists"
  exit 0
fi

if [[ -f "$PID_FILE" ]]; then
  finalizer_pid="$(tr -d '[:space:]' < "$PID_FILE")"
  if [[ "$finalizer_pid" =~ ^[0-9]+$ ]] && kill -0 "$finalizer_pid" 2>/dev/null; then
    echo "finalizer already running: $finalizer_pid"
    exit 0
  fi
fi

cd "$REPO"
nohup "$PYTHON" tools/finalize_vot_full127.py \
  --root "$ROOT" \
  --poll-seconds 60 \
  --analysis-name anchor_identity_all127_v3 \
  --expected-tracker sutrack_l384_rgbd_anchor_identity_all127 \
  --expected-manifest-sha256 d8b0b2f520e8be01321af31ad03563fa83072eb5fdd3376c2c2a2baf4f64021a \
  --candidate-name "SUTrack-L384 + anchor-specific identity-only text + safe-v1" \
  --comparison-name "SUTrack-L384 + sequence-level structured text + safe-v1 (same checkpoint/toolkit)" \
  --comparison-eao 0.7397496948296595 \
  --comparison-acc 0.8262756179006248 \
  --comparison-rob 0.8945526602400151 \
  --checkpoint /root/autodl-tmp/sutrack_assets/weights/SUTRACK_ep0180_l384.pth.tar \
  --clip-checkpoint /root/autodl-tmp/sutrack_assets/weights/ViT-L-14.pt \
  --language-manifest /root/autodl-tmp/sutrack_vot_all127_anchor_identity_v3/annotations/votrgbd2022_all127_anchor_identity.jsonl \
  --configuration experiments/sutrack/sutrack_l384_rgbd_anchor_identity_all127.yaml \
  --source-file lib/test/vot/sutrack_l384_rgbd_anchor_identity_all127.py \
  --source-file tools/build_vot_all127_anchor_identity_manifest.py \
  --source-file tools/launch_vot_all127_anchor_identity.sh \
  --source-file tools/launch_vot_all127_anchor_identity_finalizer.sh \
  --source-file /root/autodl-tmp/sutrack_rgbd_language_safe_template_vot_full127_v1/full_result.json \
  --source-file /root/autodl-tmp/sutrack_rgbd_language_safe_template_vot_full127_v1/master/analysis/full127_analysis.json \
  --source-file /root/autodl-tmp/sutrack_rgbd_language_safe_template_vot_full127_v1/merge_result.json \
  >"$LOG_FILE" 2>&1 </dev/null &
finalizer_pid=$!
temporary="$PID_FILE.tmp-$finalizer_pid"
printf '%s\n' "$finalizer_pid" > "$temporary"
mv "$temporary" "$PID_FILE"
echo "started finalizer: $finalizer_pid"
