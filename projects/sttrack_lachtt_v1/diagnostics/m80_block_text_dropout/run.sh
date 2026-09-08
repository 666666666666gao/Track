#!/bin/bash
set -u
root=/root/autodl-tmp/sttrack_m80_block_text_dropout_20260908
python=/root/autodl-tmp/envs/sttrack/bin/python
cd "$root" || exit 1
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
CUDA_VISIBLE_DEVICES=0 "$python" -u train_causal.py --arm category > training_category.log 2>&1
status=$?
printf '%s\n' "$status" > training_category.exit
if [ "$status" -ne 0 ]; then printf '%s\n' "$status" > controller.exit; exit "$status"; fi
run_eval() {
    CUDA_VISIBLE_DEVICES="$2" "$python" -u evaluate.py --condition "$1" > "eval_$1.log" 2>&1
    result=$?
    printf '%s\n' "$result" > "eval_$1.exit"
    return "$result"
}
run_eval category 0 & first=$!
run_eval empty 1 & second=$!
wait "$first"; a=$?
wait "$second"; b=$?
if [ "$a" -ne 0 ] || [ "$b" -ne 0 ]; then printf '1\n' > controller.exit; exit 1; fi
run_eval swapped 0
status=$?
if [ "$status" -ne 0 ]; then printf '%s\n' "$status" > controller.exit; exit "$status"; fi
CUDA_VISIBLE_DEVICES='' "$python" -u evaluate.py --analyze > analysis.log 2>&1
status=$?
printf '%s\n' "$status" > analysis.exit
printf '%s\n' "$status" > controller.exit
exit "$status"
