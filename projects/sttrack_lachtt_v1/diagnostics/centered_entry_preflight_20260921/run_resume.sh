#!/bin/bash
set -eu
R=/root/autodl-tmp/sttrack_centered_entry_preflight_20260921
P=/root/autodl-tmp/envs/sttrack/bin/python
C=/root/miniconda3/envs/mplt/bin/python
export CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=4
run_stage() {
    name="$1"; shift
    set +e
    "$@" > "$R/$name.log" 2>&1
    code=$?
    set -e
    echo "$code" > "$R/$name.exit"
    if [ "$code" -ne 0 ]; then exit "$code"; fi
}
for name in prepare direct_native direct_category ope_category; do
    test "$(cat "$R/$name.exit")" = 0
done
run_stage trax_category "$C" "$R/check_entries.py" client --condition category
run_stage direct_empty "$P" "$R/check_entries.py" direct --condition empty
run_stage ope_empty "$P" "$R/interface/run_semantic_ope.py" --plan "$R/empty_plan.json" --mode track
run_stage trax_empty "$C" "$R/check_entries.py" client --condition empty
run_stage verify "$P" "$R/check_entries.py" verify
