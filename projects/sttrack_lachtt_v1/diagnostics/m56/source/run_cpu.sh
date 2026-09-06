#!/usr/bin/env bash
set -u
M56_ROOT=/root/autodl-tmp/sttrack_m56_attribute_alignment_v1_20260906
M56_PYTHON=/root/autodl-tmp/envs/sttrack/bin/python
trap 'printf "%s\n" "$?" > "$M56_ROOT/cpu_controller.exit"' EXIT
"$M56_PYTHON" "$M56_ROOT/train.py" --root "$M56_ROOT" > "$M56_ROOT/train.log" 2>&1
M56_STATUS=$?
printf '%s\n' "$M56_STATUS" > "$M56_ROOT/train.exit"
if [ "$M56_STATUS" -ne 0 ]; then
    exit "$M56_STATUS"
fi
"$M56_PYTHON" "$M56_ROOT/analyze_static.py" --root "$M56_ROOT" > "$M56_ROOT/static.log" 2>&1
M56_STATUS=$?
printf '%s\n' "$M56_STATUS" > "$M56_ROOT/static.exit"
exit "$M56_STATUS"
