#!/bin/bash
cd /root/autodl-tmp/sttrack_m64_category_candidate_20260907
CUDA_VISIBLE_DEVICES='' /root/autodl-tmp/envs/sttrack/bin/python /root/autodl-tmp/sttrack_m64_category_candidate_20260907/run_inputs.py > inputs_controller.log 2>&1
