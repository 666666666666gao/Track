#!/bin/bash
cd /root/autodl-tmp/sttrack_m58_content_controls_v1_20260906
/root/autodl-tmp/envs/sttrack/bin/python -u queue_controls.py > queue.log 2>&1
status=$?
printf "%s\n" "$status" > queue.exit
exit "$status"
