#!/bin/bash
set -u
root=/root/autodl-tmp/sttrack_selected_full_evaluation_20260921
model=/root/autodl-tmp/envs/sttrack/bin/python
metric=/root/miniconda3/envs/mplt/bin/python
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$root" || exit 1

stage() {
  log="$1"; shift
  "$@" > "$root/$log.log" 2>&1
  code=$?; printf '%s\n' "$code" > "$root/$log.exit"
  if [ "$code" -ne 0 ]; then printf '%s\n' "$code" > "$root/parallel_m67_vot.exit"; exit "$code"; fi
}

stage M67/vot_bind env CUDA_VISIBLE_DEVICES='' "$model" -u prepare_full.py bind_vot --model M67
stage M67/vot_tracking env CUDA_VISIBLE_DEVICES='' "$metric" -u run_vot_failure_family_shards.py --root "$root/M67/vot/run" --poll-seconds 3600
stage M67/vot_metrics env CUDA_VISIBLE_DEVICES='' "$metric" -u analyze_full.py vot --model M67
printf '0\n' > "$root/parallel_m67_vot.exit"
