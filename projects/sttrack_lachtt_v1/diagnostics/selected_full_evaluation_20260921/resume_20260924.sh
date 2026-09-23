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
  if [ "$code" -ne 0 ]; then printf '%s\n' "$code" > "$root/resume_controller.exit"; exit "$code"; fi
}
stage migration_preflight env CUDA_VISIBLE_DEVICES=1 "$model" -u migration_preflight_20260924.py
# M67 DepthTrack Test is already complete and remains untouched.
stage M67/cdtb_track env CUDA_VISIBLE_DEVICES=1 "$model" -u interface/run_semantic_ope.py --plan "$root/M67/cdtb/plan.json" --mode track
stage M67/cdtb_metrics env CUDA_VISIBLE_DEVICES='' "$metric" -u interface/run_semantic_ope.py --plan "$root/M67/cdtb/plan.json" --mode analyze
stage M67/vot_bind env CUDA_VISIBLE_DEVICES='' "$model" -u prepare_full.py bind_vot --model M67
stage M67/vot_tracking env CUDA_VISIBLE_DEVICES='' "$metric" -u run_vot_failure_family_shards.py --root "$root/M67/vot/run" --poll-seconds 3600
stage M67/vot_metrics env CUDA_VISIBLE_DEVICES='' "$metric" -u analyze_full.py vot --model M67
for dataset in depthtrack cdtb; do
  stage "M82/${dataset}_bind" env CUDA_VISIBLE_DEVICES='' "$model" -u prepare_full.py prepare_ope --model M82 --dataset "$dataset"
  stage "M82/${dataset}_track" env CUDA_VISIBLE_DEVICES=1 "$model" -u interface/run_semantic_ope.py --plan "$root/M82/$dataset/plan.json" --mode track
  stage "M82/${dataset}_metrics" env CUDA_VISIBLE_DEVICES='' "$metric" -u interface/run_semantic_ope.py --plan "$root/M82/$dataset/plan.json" --mode analyze
done
stage M82/vot_bind env CUDA_VISIBLE_DEVICES='' "$model" -u prepare_full.py bind_vot --model M82
stage M82/vot_tracking env CUDA_VISIBLE_DEVICES='' "$metric" -u run_vot_failure_family_shards.py --root "$root/M82/vot/run" --poll-seconds 3600
stage M82/vot_metrics env CUDA_VISIBLE_DEVICES='' "$metric" -u analyze_full.py vot --model M82
stage all_results env CUDA_VISIBLE_DEVICES='' "$metric" -u analyze_full.py collect
printf '0\n' > "$root/resume_controller.exit"
