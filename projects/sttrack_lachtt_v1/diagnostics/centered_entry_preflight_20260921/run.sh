#!/bin/bash
set -eu
R=/root/autodl-tmp/sttrack_centered_entry_preflight_20260921
P=/root/autodl-tmp/envs/sttrack/bin/python
C=/root/miniconda3/envs/mplt/bin/python
export CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=4
run_stage() {
    name="$1"
    shift
    set +e
    "$@" > "$R/$name.log" 2>&1
    code=$?
    set -e
    echo "$code" > "$R/$name.exit"
    if [ "$code" -ne 0 ]; then exit "$code"; fi
}
run_stage prepare "$P" "$R/check_entries.py" prepare
run_stage direct_native "$P" "$R/check_entries.py" direct --condition native
for condition in category empty; do
    run_stage "direct_$condition" "$P" "$R/check_entries.py" direct --condition "$condition"
    run_stage "ope_$condition" "$P" "$R/interface/run_semantic_ope.py" --plan "$R/${condition}_plan.json" --mode track
    run_stage "trax_$condition" "$C" "$R/check_entries.py" client --condition "$condition"
done
run_stage verify "$P" "$R/check_entries.py" verify
