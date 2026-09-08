#!/bin/bash
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
/root/autodl-tmp/envs/sttrack/bin/python -u /root/autodl-tmp/watch_training_audits_20260908.py > /root/autodl-tmp/sttrack_training_audit_watch_20260908/watch.log 2>&1
status=$?
printf '%s\n' "$status" > /root/autodl-tmp/sttrack_training_audit_watch_20260908/watch.exit
exit "$status"
