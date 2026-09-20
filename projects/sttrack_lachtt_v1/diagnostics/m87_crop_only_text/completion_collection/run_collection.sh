#!/bin/bash
root=/root/autodl-tmp/sttrack_m87_crop_only_text_20260921/completion_collection
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
/root/autodl-tmp/envs/sttrack/bin/python -u "$root/collect_completed.py" --controller-pid 25357 > "$root/collector.log" 2>&1
status=$?
printf '%s\n' "$status" > "$root/collector.exit"
exit "$status"
