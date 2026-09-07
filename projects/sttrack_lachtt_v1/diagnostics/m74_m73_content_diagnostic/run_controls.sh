#!/bin/bash
set -u
root=/root/autodl-tmp/sttrack_m74_m73_content_diagnostic_20260907
python=/root/autodl-tmp/envs/sttrack/bin/python
script=/root/autodl-tmp/m74_m73_content_diagnostic_20260907.py
cd "$root" || exit 1
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
CUDA_VISIBLE_DEVICES='' "$python" -u "$script" eligible > eligibility.log 2>&1
status=$?; printf '%s\n' "$status" > eligibility.exit
if [ "$status" -ne 0 ]; then printf '1\n' > controller.exit; exit 1; fi
CUDA_VISIBLE_DEVICES='' "$python" -u "$script" idle > device_check.log 2>&1
status=$?; printf '%s\n' "$status" > device_check.exit
if [ "$status" -ne 0 ]; then printf '1\n' > controller.exit; exit 1; fi
CUDA_VISIBLE_DEVICES=0 "$python" -u "$script" prefix > prefix.log 2>&1
status=$?; printf '%s\n' "$status" > prefix.exit
if [ "$status" -ne 0 ]; then printf '1\n' > controller.exit; exit 1; fi
run_arm() {
    CUDA_VISIBLE_DEVICES="$2" "$python" -u "$script" "$1" > "$1.log" 2>&1
    status=$?; printf '%s\n' "$status" > "$1.exit"
    return "$status"
}
run_arm empty 0 & first=$!
run_arm swapped 1 & second=$!
wait "$first"; first_status=$?
wait "$second"; second_status=$?
if [ "$first_status" -ne 0 ] || [ "$second_status" -ne 0 ]; then printf '1\n' > controller.exit; exit 1; fi
CUDA_VISIBLE_DEVICES='' "$python" -u "$script" analyze > analysis.log 2>&1
status=$?; printf '%s\n' "$status" > analysis.exit
printf '%s\n' "$status" > controller.exit
exit "$status"
