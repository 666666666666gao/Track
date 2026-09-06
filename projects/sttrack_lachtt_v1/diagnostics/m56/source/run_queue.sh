#!/usr/bin/env bash
set -u
M56_ROOT=/root/autodl-tmp/sttrack_m56_attribute_alignment_v1_20260906
M56_GPU="$1"
M56_NOT_BEFORE="$2"
trap 'printf "%s\n" "$?" > "$M56_ROOT/queue_gpu${M56_GPU}_controller.exit"' EXIT
/root/autodl-tmp/envs/sttrack/bin/python "$M56_ROOT/queue_recursive.py" --root "$M56_ROOT" --gpu "$M56_GPU" --not-before "$M56_NOT_BEFORE" > "$M56_ROOT/queue_gpu${M56_GPU}.log" 2>&1
exit "$?"
