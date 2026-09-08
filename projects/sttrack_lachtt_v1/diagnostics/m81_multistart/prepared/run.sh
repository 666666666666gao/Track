#!/bin/bash
set -u
root=/root/autodl-tmp/sttrack_m81_multistart_20260908
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
    CUDA_VISIBLE_DEVICES="$2" "$python" -u evaluate.py --family "$1" > "eval_$1.log" 2>&1
    status=$?
    printf '%s\n' "$status" > "eval_$1.exit"
    return "$status"
}
pair_eval() {
    run_eval "$1" 0 & first=$!
    run_eval "$2" 1 & second=$!
    wait "$first"; a=$?
    wait "$second"; b=$?
    if [ "$a" -ne 0 ] || [ "$b" -ne 0 ]; then printf '1\n' > controller.exit; exit 1; fi
}
run_train category 0 & first=$!
run_train empty 1 & second=$!
wait "$first"; a=$?
wait "$second"; b=$?
if [ "$a" -ne 0 ] || [ "$b" -ne 0 ]; then printf '1\n' > controller.exit; exit 1; fi
pair_eval t0_category t0_empty_trained
pair_eval t0_empty_content t0_swapped
pair_eval multi_category multi_empty_trained
pair_eval multi_native multi_M78_category
run_eval multi_M78_empty_trained 0
status=$?
if [ "$status" -ne 0 ]; then printf '%s\n' "$status" > controller.exit; exit "$status"; fi
CUDA_VISIBLE_DEVICES='' "$python" -u evaluate.py --analyze > analysis.log 2>&1
status=$?
printf '%s\n' "$status" > analysis.exit
printf '%s\n' "$status" > controller.exit
exit "$status"
