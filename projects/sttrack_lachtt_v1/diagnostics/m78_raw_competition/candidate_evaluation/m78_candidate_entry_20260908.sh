#!/bin/bash
set -u
root=/root/autodl-tmp/sttrack_m78_raw_competition_20260908/candidate_evaluation
model=/root/autodl-tmp/envs/sttrack/bin/python
metric=/root/miniconda3/envs/mplt/bin/python
builder=/root/autodl-tmp/m78_candidate_evaluation_20260908.py
entry=/root/autodl-tmp/m78_learned_entry_parity_20260908.py
cd "$root" || exit 1
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
CUDA_VISIBLE_DEVICES='' "$model" -u "$builder" build > binding.log 2>&1
code=$?
printf '%s\n' "$code" > binding.exit
if [ "$code" -ne 0 ]; then exit "$code"; fi
CUDA_VISIBLE_DEVICES='' "$model" -u "$entry" prepare > entry_preparation.log 2>&1
code=$?
printf '%s\n' "$code" > entry_preparation.exit
if [ "$code" -ne 0 ]; then exit "$code"; fi
cd entry_parity || exit 1
CUDA_VISIBLE_DEVICES='' "$model" - <<'PY' > resource_check.log 2>&1
import json, shutil, subprocess
from pathlib import Path
queue=Path('/root/autodl-tmp/sttrack_m78_raw_competition_20260908/content_followup')
assert (queue/'controller.exit').read_text().strip()=='0'
assert json.loads((queue/'result.json').read_text())['status']=='completed_M78_fixed_Category_head_content_diagnostic'
memory=[int(x) for x in subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).splitlines()]
assert len(memory)==2 and max(memory)<500, memory
assert shutil.disk_usage('/root/autodl-tmp').free>1000000000
print('M78 training/content queue complete; both GPUs idle; disk exceeds1GB.')
PY
code=$?
printf '%s\n' "$code" > resource_check.exit
if [ "$code" -ne 0 ]; then printf '%s\n' "$code" > controller.exit; exit "$code"; fi
CUDA_VISIBLE_DEVICES=0 "$model" -u /root/autodl-tmp/sttrack_m78_raw_competition_20260908/candidate_evaluation/interface/run_semantic_ope.py --plan "$root/entry_parity/ope_plan.json" --mode track > ope.log 2>&1 & ope=$!
CUDA_VISIBLE_DEVICES='' "$metric" -u "$entry" client > trax.log 2>&1 & trax=$!
wait "$ope"; first=$?
printf '%s\n' "$first" > ope.exit
wait "$trax"; second=$?
printf '%s\n' "$second" > trax.exit
if [ "$first" -ne 0 ] || [ "$second" -ne 0 ]; then printf '1\n' > controller.exit; exit 1; fi
CUDA_VISIBLE_DEVICES='' "$metric" -u "$entry" verify > verification.log 2>&1
code=$?
printf '%s\n' "$code" > verification.exit
printf '%s\n' "$code" > controller.exit
exit "$code"
