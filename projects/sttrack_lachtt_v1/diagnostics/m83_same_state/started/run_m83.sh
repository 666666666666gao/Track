#!/bin/bash
set -u
cd /root/autodl-tmp/sttrack_m83_same_state_20260920
test "$(cat preflight.exit)" = 0 || exit 1
CUDA_VISIBLE_DEVICES=0 /root/autodl-tmp/envs/sttrack/bin/python m83_same_state.py > replay.log 2>&1
rc=$?
printf '%s\n' "$rc" > replay.exit
if [ "$rc" -eq 0 ]; then
  /root/autodl-tmp/envs/sttrack/bin/python analyze_m83.py > analysis.log 2>&1
  rc=$?
  printf '%s\n' "$rc" > analysis.exit
fi
printf '%s\n' "$rc" > controller.exit
exit "$rc"
