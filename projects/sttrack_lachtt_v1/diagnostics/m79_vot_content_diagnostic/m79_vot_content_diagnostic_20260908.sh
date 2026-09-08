#!/bin/bash
set -u
root=/root/autodl-tmp/sttrack_m79_vot_content_diagnostic_20260908
model=/root/autodl-tmp/envs/sttrack/bin/python
metric=/root/miniconda3/envs/mplt/bin/python
script=/root/autodl-tmp/m79_vot_content_diagnostic_20260908.py
cd "$root" || exit 1
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
CUDA_VISIBLE_DEVICES='' "$model" -u "$script" check > binding.log 2>&1
code=$?
printf '%s\n' "$code" > binding.exit
if [ "$code" -ne 0 ]; then printf '%s\n' "$code" > controller.exit; exit "$code"; fi
run_arm() {
    arm="$1"
    CUDA_VISIBLE_DEVICES='' "$metric" -u run_vot_failure_family_shards.py --root "$root/$arm/run" --poll-seconds 240 > "$arm-tracking.log" 2>&1
    code=$?
    printf '%s\n' "$code" > "${arm}_tracking.exit"
    if [ "$code" -eq 0 ]; then
        CUDA_VISIBLE_DEVICES='' "$metric" -u "$script" "$arm" > "$arm-analysis.log" 2>&1
        code=$?
        printf '%s\n' "$code" > "${arm}_analysis.exit"
    fi
    printf '%s\n' "$code" > "$arm.exit"
    return "$code"
}
run_arm empty & first=$!
run_arm swapped & second=$!
wait "$first"; first_status=$?
wait "$second"; second_status=$?
if [ "$first_status" -ne 0 ] || [ "$second_status" -ne 0 ]; then printf '1\n' > controller.exit; exit 1; fi
CUDA_VISIBLE_DEVICES='' "$metric" -u "$script" analyze > analysis.log 2>&1
code=$?
printf '%s\n' "$code" > analysis.exit
printf '%s\n' "$code" > controller.exit
exit "$code"
