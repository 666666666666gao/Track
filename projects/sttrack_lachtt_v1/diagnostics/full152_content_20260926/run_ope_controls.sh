#!/usr/bin/env bash
set -u

root=/root/autodl-tmp/sttrack_full152_content_20260926
main=/root/autodl-tmp/sttrack_full152_evaluation_20260925
model_python=/root/autodl-tmp/envs/sttrack/bin/python
metric_python=/root/miniconda3/envs/mplt/bin/python
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

test "$(cat "$main/full_eval.exit")" = 0 || exit 1
test "$(cat "$main/M82/vot_metrics.exit")" = 0 || exit 1
cd "$root" || exit 1
mkdir -p logs

for model in M82 M67; do
  for variant in empty swapped; do
    for dataset in depthtrack cdtb; do
      "$model_python" -u prepare_content.py --model "$model" --dataset "$dataset" --variant "$variant" \
        > "logs/${model}_${dataset}_${variant}_prepare.log" 2>&1 || exit 1
    done

    CUDA_VISIBLE_DEVICES=0 "$model_python" -u "$main/interface/run_semantic_ope.py" \
      --plan "$root/$model/depthtrack/$variant/plan.json" --mode track \
      > "logs/${model}_depthtrack_${variant}_track.log" 2>&1 &
    depth_pid=$!
    CUDA_VISIBLE_DEVICES=1 "$model_python" -u "$main/interface/run_semantic_ope.py" \
      --plan "$root/$model/cdtb/$variant/plan.json" --mode track \
      > "logs/${model}_cdtb_${variant}_track.log" 2>&1 &
    cdtb_pid=$!
    wait "$depth_pid"; depth_status=$?
    wait "$cdtb_pid"; cdtb_status=$?
    printf '%s\n' "$depth_status" > "logs/${model}_depthtrack_${variant}_track.exit"
    printf '%s\n' "$cdtb_status" > "logs/${model}_cdtb_${variant}_track.exit"
    if [ "$depth_status" -ne 0 ] || [ "$cdtb_status" -ne 0 ]; then exit 1; fi

    for dataset in depthtrack cdtb; do
      "$metric_python" -u "$main/interface/run_semantic_ope.py" \
        --plan "$root/$model/$dataset/$variant/plan.json" --mode analyze \
        > "logs/${model}_${dataset}_${variant}_metrics.log" 2>&1 || exit 1
    done
  done
done

printf '0\n' > "$root/controls.exit"
