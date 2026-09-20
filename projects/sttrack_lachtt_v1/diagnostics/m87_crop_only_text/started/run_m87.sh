#!/bin/bash
set -u
root=/root/autodl-tmp/sttrack_m87_crop_only_text_20260921
python=/root/autodl-tmp/envs/sttrack/bin/python
cd "$root" || exit 1
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
CUDA_VISIBLE_DEVICES=0 "$python" -u train_causal.py --arm category > training_category.log 2>&1
status=$?
printf '%s\n' "$status" > training_category.exit
if [ "$status" -ne 0 ]; then printf '1\n' > controller.exit; exit 1; fi
run_eval() {
    CUDA_VISIBLE_DEVICES="$2" "$python" -u run_recursive.py --arm "$1" > "$1"_recursive.log 2>&1
    status=$?
    printf '%s\n' "$status" > "$1"_recursive.exit
    return "$status"
}
run_eval category 0 & first=$!
run_eval category_empty 1 & second=$!
wait "$first"; a=$?
wait "$second"; b=$?
if [ "$a" -ne 0 ] || [ "$b" -ne 0 ]; then printf '1\n' > controller.exit; exit 1; fi
run_eval category_old 0 & first=$!
run_eval category_swapped 1 & second=$!
wait "$first"; a=$?
wait "$second"; b=$?
if [ "$a" -ne 0 ] || [ "$b" -ne 0 ]; then printf '1\n' > controller.exit; exit 1; fi
CUDA_VISIBLE_DEVICES='' "$python" -u run_recursive.py --analyze > recursive_analysis.log 2>&1
status=$?
printf '%s\n' "$status" > recursive_analysis.exit
printf '%s\n' "$status" > controller.exit
exit "$status"
