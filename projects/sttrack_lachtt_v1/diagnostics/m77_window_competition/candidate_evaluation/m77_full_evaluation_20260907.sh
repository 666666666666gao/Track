#!/bin/bash
set -u
root=/root/autodl-tmp/sttrack_m77_window_competition_20260907/candidate_evaluation
model=/root/autodl-tmp/envs/sttrack/bin/python
metric=/root/miniconda3/envs/mplt/bin/python
caption=/home/qwen25_env/bin/python
helper=/root/autodl-tmp/m77_full_preparation_20260907.py
vot=/root/autodl-tmp/m77_full_vot_20260907.py
generator=/root/autodl-tmp/sttrack_m58_initialization_generator_20260906/initialization_captions.py
cd "$root" || exit 1
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
CUDA_VISIBLE_DEVICES='' "$model" -u "$helper" eligible
status=$?
if [ "$status" -ne 0 ]; then exit "$status"; fi
mkdir full_evaluation || exit 1
cd full_evaluation || exit 1
step() {
    name="$1"; shift
    "$@" > "$name.log" 2>&1
    status=$?; printf '%s\n' "$status" > "$name.exit"
    if [ "$status" -ne 0 ]; then printf '%s\n' "$status" > controller.exit; exit "$status"; fi
}
step resource_check env CUDA_VISIBLE_DEVICES='' "$model" - <<'PY'
import shutil,subprocess
used=[int(x) for x in subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).splitlines()]
assert len(used)==2 and max(used)<500
assert shutil.disk_usage('/root/autodl-tmp').free>1000000000
print('Both GPUs idle and disk exceeds1GB.')
PY
for dataset in depthtrack cdtb; do
    step "$dataset.prepare" env CUDA_VISIBLE_DEVICES='' "$model" -u "$helper" prepare_ope --dataset "$dataset"
    step "$dataset.generate" env CUDA_VISIBLE_DEVICES=0 "$caption" -u "$generator" generate --plan "$root/full_evaluation/$dataset/captions/plan.json"
    step "$dataset.encode" env CUDA_VISIBLE_DEVICES=0 "$model" -u "$generator" encode --plan "$root/full_evaluation/$dataset/captions/plan.json"
    step "$dataset.bank" env CUDA_VISIBLE_DEVICES='' "$model" -u "$helper" bank_ope --dataset "$dataset"
    step "$dataset.track" env CUDA_VISIBLE_DEVICES=0 "$model" -u "$root/interface/run_semantic_ope.py" --plan "$root/full_evaluation/$dataset/plan.json" --mode track
    step "$dataset.metric" env CUDA_VISIBLE_DEVICES='' "$metric" -u "$root/interface/run_semantic_ope.py" --plan "$root/full_evaluation/$dataset/plan.json" --mode analyze
done
step vot.export env CUDA_VISIBLE_DEVICES='' "$model" -u "$helper" export_vot
step vot.prepare env CUDA_VISIBLE_DEVICES='' "$model" -u "$vot" prepare_captions
step vot.generate env CUDA_VISIBLE_DEVICES=0 "$caption" -u "$generator" generate --plan "$root/full_evaluation/vot/remaining_captions/plan.json"
step vot.encode env CUDA_VISIBLE_DEVICES=0 "$model" -u "$generator" encode --plan "$root/full_evaluation/vot/remaining_captions/plan.json"
step vot.bind env CUDA_VISIBLE_DEVICES='' "$model" -u "$vot" bind_and_prepare_run
step vot.tracking env CUDA_VISIBLE_DEVICES='' "$metric" -u "$root/full_evaluation/vot/run_vot_failure_family_shards.py" --root "$root/full_evaluation/vot/run" --poll-seconds 240
printf '0\n' > vot/tracking.exit
step vot.analysis env CUDA_VISIBLE_DEVICES='' "$metric" -u "$vot" analyze
step verification env CUDA_VISIBLE_DEVICES='' "$metric" -u /root/autodl-tmp/verify_m77_three_datasets_20260907.py
printf '0\n' > controller.exit
