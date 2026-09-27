#!/usr/bin/env bash
set -euo pipefail

root=/root/autodl-tmp/sttrack_m89_vot_readout_20260928
content=/root/autodl-tmp/sttrack_m89_content_20260927
model=/root/autodl-tmp/envs/sttrack/bin/python
metric=/root/miniconda3/envs/mplt/bin/python
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

test "$(cat "$content/complete.exit")" = 0
CUDA_VISIBLE_DEVICES=0 "$model" "$root/readout.py" --shard 0 --preflight \
    >"$root/preflight.log" 2>&1
CUDA_VISIBLE_DEVICES=0 "$model" "$root/readout.py" --shard 0 \
    >"$root/shard0.log" 2>&1 &
pid0=$!
CUDA_VISIBLE_DEVICES=1 "$model" "$root/readout.py" --shard 1 \
    >"$root/shard1.log" 2>&1 &
pid1=$!
set +e
wait "$pid0"; rc0=$?
wait "$pid1"; rc1=$?
set -e
printf '%s\n' "$rc0" >"$root/shard0.exit"
printf '%s\n' "$rc1" >"$root/shard1.exit"
test "$rc0" = 0
test "$rc1" = 0
"$metric" "$root/analyze.py" >"$root/analysis.log" 2>&1
printf '0\n' >"$root/complete.exit"
