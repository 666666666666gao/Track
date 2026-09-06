#!/bin/bash
cd /root/autodl-tmp/sttrack_m64_category_candidate_20260907
CUDA_VISIBLE_DEVICES='' /root/autodl-tmp/envs/sttrack/bin/python /root/autodl-tmp/m64_vot_low22_20260907.py check > low22_binding.log 2>&1
code=$?
printf '%s\n' "$code" > low22_binding.exit
if [ "$code" -eq 0 ]; then
CUDA_VISIBLE_DEVICES='' /root/miniconda3/envs/mplt/bin/python run_vot_failure_family_shards.py --root /root/autodl-tmp/sttrack_m64_category_candidate_20260907/low22_run --poll-seconds 240 > low22_tracking.log 2>&1
code=$?
printf '%s\n' "$code" > low22_tracking.exit
fi
if [ "$code" -eq 0 ]; then
CUDA_VISIBLE_DEVICES='' /root/miniconda3/envs/mplt/bin/python /root/autodl-tmp/m64_vot_low22_20260907.py analyze > low22_analysis.log 2>&1
code=$?
printf '%s\n' "$code" > low22_analysis.exit
fi
printf '%s\n' "$code" > low22_controller.exit
exit "$code"
