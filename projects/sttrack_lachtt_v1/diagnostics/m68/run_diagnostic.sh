#!/usr/bin/env bash
set -u
cd /root/autodl-tmp/sttrack_m68_reported_confidence_20260907
trap 'rc=$?; printf "%s\n" "$rc" > controller.exit' EXIT
/root/autodl-tmp/envs/sttrack/bin/python /root/autodl-tmp/m68_reported_confidence_20260907.py eligible > eligibility.log 2>&1 || exit $?
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
while read -r value; do [ "$value" -lt 500 ] || exit 70; done <<< "$used"
CUDA_VISIBLE_DEVICES=0 /root/autodl-tmp/envs/sttrack/bin/python /root/autodl-tmp/m68_reported_confidence_20260907.py control > control.log 2>&1 &
p0=$!
CUDA_VISIBLE_DEVICES=1 /root/autodl-tmp/envs/sttrack/bin/python /root/autodl-tmp/m68_reported_confidence_20260907.py support > support.log 2>&1 &
p1=$!
wait "$p0"; r0=$?; printf "%s\n" "$r0" > control.exit
wait "$p1"; r1=$?; printf "%s\n" "$r1" > support.exit
[ "$r0" -eq 0 ] && [ "$r1" -eq 0 ] || exit 71
/root/miniconda3/envs/mplt/bin/python /root/autodl-tmp/m68_reported_confidence_20260907.py analyze > analysis.log 2>&1
rc=$?; printf "%s\n" "$rc" > analysis.exit
exit "$rc"
