#!/usr/bin/env bash
set -euo pipefail

root=/root/autodl-tmp/sttrack_native_candidate_preservation_20260926
readout=/root/autodl-tmp/sttrack_m89_vot_readout_20260928
python=/root/autodl-tmp/envs/sttrack/bin/python
metric=/root/miniconda3/envs/mplt/bin/python

while [ ! -f "$readout/complete.exit" ]; do
    sleep 300
done
test "$(cat "$readout/complete.exit")" = 0
mkdir "$root/probe"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

for name in old_a zero old_b; do
    mode=old
    if [ "$name" = zero ]; then mode=zero; fi
    CUDA_VISIBLE_DEVICES=0 "$python" -u "$root/probe_zero_prefix.py" \
        --mode "$mode" --output "$root/probe/$name.json" \
        >"$root/probe/$name.log" 2>&1
done
"$metric" "$root/compare_zero_prefix.py" >"$root/probe/compare.log" 2>&1
printf '0\n' >"$root/probe/complete.exit"
