#!/bin/bash
CUDA_VISIBLE_DEVICES=1 /root/autodl-tmp/envs/sttrack/bin/python -u /root/autodl-tmp/sttrack_m57_initial_binding_v1_20260906/collect_initial.py --spec /root/autodl-tmp/sttrack_m57_initial_binding_v1_20260906/collection_spec.json > /root/autodl-tmp/sttrack_m57_initial_binding_v1_20260906/collection.log 2>&1 &
child=$!
printf "%s\n" "$child" > /root/autodl-tmp/sttrack_m57_initial_binding_v1_20260906/collection.pid
wait "$child"
status=$?
printf "%s\n" "$status" > /root/autodl-tmp/sttrack_m57_initial_binding_v1_20260906/collection.exit
exit "$status"
