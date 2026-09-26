#!/usr/bin/env bash
set -euo pipefail
SOURCE=/root/autodl-tmp/sttrack_native_candidate_preservation_20260926
ROOT=/root/autodl-tmp/sttrack_m89_evaluation_20260926
OLD=/root/autodl-tmp/sttrack_full152_evaluation_20260925
MODEL=/root/autodl-tmp/envs/sttrack/bin/python
CPU=/root/miniconda3/envs/mplt/bin/python
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
test "$(cat "$SOURCE/training.exit")" = 0
"$MODEL" "$SOURCE/prepare_evaluation.py" bind
stage() {
    local output=$1
    shift
    local rc=0
    "$@" >"$output.log" 2>&1 || rc=$?
    echo "$rc" >"$output.exit"
    return "$rc"
}
for arm in control candidate; do
    stage "$ROOT/$arm/depthtrack_tracking" env CUDA_VISIBLE_DEVICES=0 "$MODEL" "$OLD/interface/run_semantic_ope.py" --plan "$ROOT/$arm/depthtrack/plan.json" --mode track &
    depth_pid=$!
    stage "$ROOT/$arm/cdtb_tracking" env CUDA_VISIBLE_DEVICES=1 "$MODEL" "$OLD/interface/run_semantic_ope.py" --plan "$ROOT/$arm/cdtb/plan.json" --mode track &
    cdtb_pid=$!
    depth_rc=0; cdtb_rc=0
    wait "$depth_pid" || depth_rc=$?
    wait "$cdtb_pid" || cdtb_rc=$?
    test "$depth_rc" = 0
    test "$cdtb_rc" = 0
    for dataset in depthtrack cdtb; do
        stage "$ROOT/$arm/${dataset}_analysis" "$CPU" "$OLD/interface/run_semantic_ope.py" --plan "$ROOT/$arm/$dataset/plan.json" --mode analyze
    done
    stage "$ROOT/$arm/vot_binding" "$CPU" "$SOURCE/prepare_evaluation.py" bind-vot --model "$arm"
    stage "$ROOT/$arm/vot_tracking" "$CPU" "$OLD/run_vot_failure_family_shards.py" --root "$ROOT/$arm/vot/run" --poll-seconds 300
    stage "$ROOT/$arm/vot_analysis" "$CPU" "$SOURCE/analyze_evaluation.py" vot --model "$arm"
done
stage "$ROOT/collection" "$CPU" "$SOURCE/analyze_evaluation.py" collect
echo 0 >"$ROOT/evaluation.exit"
