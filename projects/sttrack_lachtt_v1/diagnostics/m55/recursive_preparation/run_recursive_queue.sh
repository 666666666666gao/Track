#!/usr/bin/env bash
set -u
M55_ROOT=/root/autodl-tmp/sttrack_m55_tsg_direction_v2_20260906
M56_ROOT=/root/autodl-tmp/sttrack_m56_attribute_alignment_v1_20260906
M55_GPU="$1"
M55_NOT_BEFORE="$2"
trap 'printf "%s\n" "$?" > "$M55_ROOT/recursive_queue_gpu${M55_GPU}_controller.exit"' EXIT
/root/autodl-tmp/envs/sttrack/bin/python "$M55_ROOT/queue_recursive.py" --root "$M55_ROOT" --predecessor "$M56_ROOT" --gpu "$M55_GPU" --not-before "$M55_NOT_BEFORE" > "$M55_ROOT/recursive_queue_gpu${M55_GPU}.log" 2>&1
exit "$?"
