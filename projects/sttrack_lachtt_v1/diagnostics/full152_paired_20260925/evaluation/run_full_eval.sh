#!/usr/bin/env bash
set -u

root=/root/autodl-tmp/sttrack_full152_evaluation_20260925
train=/root/autodl-tmp/sttrack_full152_paired_20260925
model=/root/autodl-tmp/envs/sttrack/bin/python
metric=/root/miniconda3/envs/mplt/bin/python
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$root" || exit 1

test "$(cat "$train/M67/training/train.exit")" = 0 || exit 1
test "$(cat "$train/M82/training/train.exit")" = 0 || exit 1

stage() {
  label="$1"; shift
  "$@" > "$root/$label.log" 2>&1
  status=$?
  printf '%s\n' "$status" > "$root/$label.exit"
  if [ "$status" -ne 0 ]; then
    printf '%s\n' "$status" > "$root/full_eval.exit"
    exit "$status"
  fi
}

stage bind_models "$model" -u prepare_full.py bind_models

for name in M67 M82; do
  stage "$name/depthtrack_bind" "$model" -u prepare_full.py prepare_ope --model "$name" --dataset depthtrack
  stage "$name/cdtb_bind" "$model" -u prepare_full.py prepare_ope --model "$name" --dataset cdtb

  env CUDA_VISIBLE_DEVICES=0 "$model" -u interface/run_semantic_ope.py \
    --plan "$root/$name/depthtrack/plan.json" --mode track > "$root/$name/depthtrack_track.log" 2>&1 &
  depth_pid=$!
  env CUDA_VISIBLE_DEVICES=1 "$model" -u interface/run_semantic_ope.py \
    --plan "$root/$name/cdtb/plan.json" --mode track > "$root/$name/cdtb_track.log" 2>&1 &
  cdtb_pid=$!
  wait "$depth_pid"; depth_status=$?
  printf '%s\n' "$depth_status" > "$root/$name/depthtrack_track.exit"
  wait "$cdtb_pid"; cdtb_status=$?
  printf '%s\n' "$cdtb_status" > "$root/$name/cdtb_track.exit"
  if [ "$depth_status" -ne 0 ] || [ "$cdtb_status" -ne 0 ]; then
    printf '1\n' > "$root/full_eval.exit"
    exit 1
  fi

  stage "$name/depthtrack_metrics" "$metric" -u interface/run_semantic_ope.py \
    --plan "$root/$name/depthtrack/plan.json" --mode analyze
  stage "$name/cdtb_metrics" "$metric" -u interface/run_semantic_ope.py \
    --plan "$root/$name/cdtb/plan.json" --mode analyze

  stage "$name/vot_bind" "$model" -u prepare_full.py bind_vot --model "$name"
  stage "$name/vot_tracking" "$metric" -u run_vot_failure_family_shards.py \
    --root "$root/$name/vot/run" --poll-seconds 300
  stage "$name/vot_metrics" "$metric" -u analyze_full.py vot --model "$name"
done

stage all_results "$metric" -u analyze_full.py collect
printf '0\n' > "$root/full_eval.exit"
