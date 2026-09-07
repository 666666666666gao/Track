#!/bin/bash
set -u
root=/root/autodl-tmp/sttrack_m73_paired_lexical_replication_20260907/seed2028
python=/root/autodl-tmp/envs/sttrack/bin/python
cd "$root" || exit 1
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
run_train() {
    CUDA_VISIBLE_DEVICES="$2" "$python" -u train_causal.py --arm "$1" > "training_$1.log" 2>&1
    status=$?
    printf '%s\n' "$status" > "training_$1.exit"
    return "$status"
}
run_eval() {
    CUDA_VISIBLE_DEVICES="$2" "$python" -u run_recursive.py --arm "$1" > "$1_recursive.log" 2>&1
    status=$?
    printf '%s\n' "$status" > "$1_recursive.exit"
    return "$status"
}
run_train empty 0 & control_pid=$!
run_train category 1 & support_pid=$!
wait "$control_pid"; control_status=$?
wait "$support_pid"; support_status=$?
if [ "$control_status" -ne 0 ] || [ "$support_status" -ne 0 ]; then
    printf '1\n' > controller.exit
    exit 1
fi
run_eval empty 0 & control_pid=$!
run_eval category 1 & support_pid=$!
wait "$control_pid"; control_status=$?
wait "$support_pid"; support_status=$?
if [ "$control_status" -ne 0 ] || [ "$support_status" -ne 0 ]; then
    printf '1\n' > controller.exit
    exit 1
fi
CUDA_VISIBLE_DEVICES='' "$python" -u run_recursive.py --analyze > recursive_analysis.log 2>&1
status=$?
printf '%s\n' "$status" > recursive_analysis.exit
printf '%s\n' "$status" > controller.exit
exit "$status"
