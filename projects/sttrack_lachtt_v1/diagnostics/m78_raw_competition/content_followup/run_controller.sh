#!/bin/bash
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
/root/autodl-tmp/envs/sttrack/bin/python -u /root/autodl-tmp/m78_content_followup_20260908.py controller
status=$?
printf "%s\n" "$status" > /root/autodl-tmp/sttrack_m78_raw_competition_20260908/content_followup/controller.exit
exit "$status"
