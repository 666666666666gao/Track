#!/bin/bash
set -u
cd /root/autodl-tmp/sttrack_m73_paired_lexical_replication_20260907/content_counterfactuals || exit 1
python=/root/autodl-tmp/envs/sttrack/bin/python
script=/root/autodl-tmp/m73_content_counterfactuals_20260907.py
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
run_step() {
    CUDA_VISIBLE_DEVICES="$2" "$python" -u "$script" "$1" > "$1.log" 2>&1
    status=$?; printf '%s\n' "$status" > "$1.exit"
    return "$status"
}
for action in activate idle; do
    run_step "$action" '' || { printf '1\n' > controller.exit; exit 1; }
done
run_step prefix 0 || { printf '1\n' > controller.exit; exit 1; }
run_step empty 0 & first=$!
run_step swapped 1 & second=$!
wait "$first"; first_status=$?
wait "$second"; second_status=$?
if [ "$first_status" -ne 0 ] || [ "$second_status" -ne 0 ]; then printf '1\n' > controller.exit; exit 1; fi
run_step analyze ''; status=$?
printf '%s\n' "$status" > controller.exit
exit "$status"
