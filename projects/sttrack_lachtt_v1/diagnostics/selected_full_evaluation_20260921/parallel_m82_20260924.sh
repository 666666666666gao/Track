#!/bin/bash
set -u
root=/root/autodl-tmp/sttrack_selected_full_evaluation_20260921
model=/root/autodl-tmp/envs/sttrack/bin/python
metric=/root/miniconda3/envs/mplt/bin/python
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$root" || exit 1

test "$(cat M67/cdtb_metrics.exit)" = 0 || exit 1
test "$(cat M67/vot_metrics.exit)" = 0 || exit 1
test "$(cat parallel_m67_vot.exit)" = 0 || exit 1
test -f M67/cdtb/predictions/metrics.json || exit 1
test -f M67/vot/result.json || exit 1

stage() {
  log="$1"; shift
  "$@" > "$root/$log.log" 2>&1
  code=$?; printf '%s\n' "$code" > "$root/$log.exit"
  if [ "$code" -ne 0 ]; then printf '%s\n' "$code" > "$root/parallel_m82.exit"; exit "$code"; fi
}

stage M82/depthtrack_bind env CUDA_VISIBLE_DEVICES='' "$model" -u prepare_full.py prepare_ope --model M82 --dataset depthtrack
stage M82/cdtb_bind env CUDA_VISIBLE_DEVICES='' "$model" -u prepare_full.py prepare_ope --model M82 --dataset cdtb

env CUDA_VISIBLE_DEVICES=0 "$model" -u interface/run_semantic_ope.py --plan "$root/M82/depthtrack/plan.json" --mode track > M82/depthtrack_track.log 2>&1 &
depth_pid=$!
env CUDA_VISIBLE_DEVICES=1 "$model" -u interface/run_semantic_ope.py --plan "$root/M82/cdtb/plan.json" --mode track > M82/cdtb_track.log 2>&1 &
cdtb_pid=$!
wait "$depth_pid"; depth_code=$?; printf '%s\n' "$depth_code" > M82/depthtrack_track.exit
wait "$cdtb_pid"; cdtb_code=$?; printf '%s\n' "$cdtb_code" > M82/cdtb_track.exit
if [ "$depth_code" -ne 0 ] || [ "$cdtb_code" -ne 0 ]; then printf '1\n' > parallel_m82.exit; exit 1; fi

stage M82/depthtrack_metrics env CUDA_VISIBLE_DEVICES='' "$metric" -u interface/run_semantic_ope.py --plan "$root/M82/depthtrack/plan.json" --mode analyze
stage M82/cdtb_metrics env CUDA_VISIBLE_DEVICES='' "$metric" -u interface/run_semantic_ope.py --plan "$root/M82/cdtb/plan.json" --mode analyze
stage M82/vot_bind env CUDA_VISIBLE_DEVICES='' "$model" -u prepare_full.py bind_vot --model M82
stage M82/vot_tracking env CUDA_VISIBLE_DEVICES='' "$metric" -u run_vot_failure_family_shards.py --root "$root/M82/vot/run" --poll-seconds 3600
stage M82/vot_metrics env CUDA_VISIBLE_DEVICES='' "$metric" -u analyze_full.py vot --model M82
stage all_results env CUDA_VISIBLE_DEVICES='' "$metric" -u analyze_full.py collect
stage full_comparison env CUDA_VISIBLE_DEVICES='' "$metric" -u export_full_comparison_20260924.py
printf '0\n' > "$root/parallel_m82.exit"
