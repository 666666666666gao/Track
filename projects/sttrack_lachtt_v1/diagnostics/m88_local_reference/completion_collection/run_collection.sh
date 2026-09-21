#!/bin/bash
root=/root/autodl-tmp/sttrack_m88_local_reference_20260921/completion_collection
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
/root/autodl-tmp/envs/sttrack/bin/python -u "$root/collect_completed.py" --controller-pid 48174 > "$root/collector.log" 2>&1
status=$?
printf '%s\n' "$status" > "$root/collector.exit"
exit "$status"
