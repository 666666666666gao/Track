#!/bin/bash
cd /root/autodl-tmp/sttrack_m63_location_write_factorial_20260907
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /root/autodl-tmp/envs/sttrack/bin/python /root/autodl-tmp/m63_location_write_factorial_20260907.py queue > queue.log 2>&1
code=$?
printf '%s\n' "$code" > controller.exit
exit "$code"
