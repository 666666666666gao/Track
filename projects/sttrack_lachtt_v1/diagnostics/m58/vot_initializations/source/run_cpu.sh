#!/bin/bash
cd /root/autodl-tmp/sttrack_m58_vot_initialization_export_20260906
export CUDA_VISIBLE_DEVICES=''
/root/miniconda3/envs/mplt/bin/python -u export_initializations.py --manifest /root/autodl-tmp/sutrack_vot_low22_anchor_identity_v1/run/shard_manifest.json --expected-sha 600b1ebb8b0c2f69b831f954e907e63709fd69afb7ea94c5b58e8c7408a29eed --output low22_inputs > export.log 2>&1
code=$?
printf '%s\n' "$code" > export.exit
if [ "$code" != 0 ]; then exit "$code"; fi
/root/miniconda3/envs/mplt/bin/python -u check_scheduler.py > scheduler.log 2>&1
code=$?
printf '%s\n' "$code" > scheduler.exit
exit "$code"
