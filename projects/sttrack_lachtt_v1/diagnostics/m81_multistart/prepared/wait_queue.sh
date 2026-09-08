#!/bin/bash
/root/miniconda3/envs/mplt/bin/python -u /root/autodl-tmp/queue_m81_after_m80_20260908.py > /root/autodl-tmp/sttrack_m81_multistart_20260908/queue.log 2>&1
status=$?
printf "%s\n" "$status" > /root/autodl-tmp/sttrack_m81_multistart_20260908/queue.exit
exit "$status"
