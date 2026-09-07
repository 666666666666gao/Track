#!/bin/bash
set -u
cd /root/autodl-tmp/sttrack_m74_m73_content_diagnostic_20260907 || exit 1
while [ ! -f controller.exit ]; do sleep 240; done
parent_status=$(cat controller.exit)
if [ "$parent_status" -ne 0 ]; then printf '%s\n' "$parent_status" > completion_audit_runner/controller.exit; exit "$parent_status"; fi
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /root/autodl-tmp/envs/sttrack/bin/python -u /root/autodl-tmp/audit_m74_completed_20260907.py > completion_audit_runner/audit.log 2>&1
status=$?
printf '%s\n' "$status" > completion_audit_runner/controller.exit
exit "$status"
