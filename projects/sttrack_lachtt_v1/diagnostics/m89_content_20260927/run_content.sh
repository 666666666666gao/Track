#!/usr/bin/env bash
set -euo pipefail

root=/root/autodl-tmp/sttrack_m89_content_20260927
interface=/root/autodl-tmp/sttrack_full152_evaluation_20260925/interface/run_semantic_ope.py
model=/root/autodl-tmp/envs/sttrack/bin/python
metric=/root/miniconda3/envs/mplt/bin/python
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

for variant in empty swapped; do
    for dataset in depthtrack cdtb; do
        "$model" "$root/prepare_content.py" --dataset "$dataset" --variant "$variant" \
            >"$root/${dataset}_${variant}_prepare.log" 2>&1
    done
    CUDA_VISIBLE_DEVICES=0 "$model" "$interface" --plan "$root/depthtrack/$variant/plan.json" --mode track \
        >"$root/depthtrack_${variant}_track.log" 2>&1 &
    depth_pid=$!
    CUDA_VISIBLE_DEVICES=1 "$model" "$interface" --plan "$root/cdtb/$variant/plan.json" --mode track \
        >"$root/cdtb_${variant}_track.log" 2>&1 &
    cdtb_pid=$!
    wait "$depth_pid"
    wait "$cdtb_pid"
    for dataset in depthtrack cdtb; do
        "$metric" "$interface" --plan "$root/$dataset/$variant/plan.json" --mode analyze \
            >"$root/${dataset}_${variant}_analysis.log" 2>&1
    done
done

printf '0\n' >"$root/complete.exit"
