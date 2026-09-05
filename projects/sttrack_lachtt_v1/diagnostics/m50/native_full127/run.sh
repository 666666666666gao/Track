#!/bin/bash
set -u
cd /root/autodl-tmp/sttrack_default_full127_v1_20260905
printf '%s\n' "$$" > controller.pid
/root/miniconda3/envs/mplt/bin/python tools/run_vot_failure_family_shards.py --root /root/autodl-tmp/sttrack_default_full127_v1_20260905/run --poll-seconds 240 > evaluate.log 2>&1
code=$?
printf '%s\n' "$code" > evaluate.exit
if [ "$code" -eq 0 ]; then
/root/miniconda3/envs/mplt/bin/python finalize_default_full127.py > finalize.log 2>&1
code=$?
printf '%s\n' "$code" > finalize.exit
fi
printf '%s\n' "$code" > controller.exit
exit "$code"
