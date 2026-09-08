#!/bin/bash
set -u
root=/root/autodl-tmp/sttrack_m81_multistart_20260908
generator=/root/autodl-tmp/sttrack_m58_initialization_generator_20260906/initialization_captions.py
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
CUDA_VISIBLE_DEVICES=1 /home/qwen25_env/bin/python -u "$generator" generate --plan "$root/captions/plan.json" > "$root/caption_generate.log" 2>&1
status=$?
printf '%s\n' "$status" > "$root/caption_generate.exit"
if [ "$status" -ne 0 ]; then printf '%s\n' "$status" > "$root/caption_controller.exit"; exit "$status"; fi
CUDA_VISIBLE_DEVICES='' /root/autodl-tmp/envs/sttrack/bin/python -u "$generator" encode --plan "$root/captions/plan.json" > "$root/caption_encode.log" 2>&1
status=$?
printf '%s\n' "$status" > "$root/caption_encode.exit"
printf '%s\n' "$status" > "$root/caption_controller.exit"
exit "$status"
