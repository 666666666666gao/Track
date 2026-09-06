#!/bin/bash
set -uo pipefail
root=/root/autodl-tmp/sttrack_m55_tsg_direction_v2_20260906
native=/root/autodl-tmp/sttrack_default_full127_v1_20260905
cd "$root"
printf '%s\n' "$$" > clone_queue.pid
while [ ! -f "$native/controller.exit" ]; do
  date --iso-8601=seconds > clone_queue_last_check.txt
  sleep 240
done
if [ "$(cat "$native/controller.exit")" -ne 0 ]; then
  printf 'Native full127 controller exited nonzero; clone was not launched.\n' > clone_queue.log
  printf '1\n' > clone_queue.exit
  exit 1
fi
used=$(nvidia-smi -i 1 --query-gpu=memory.used --format=csv,noheader,nounits)
if [ "$used" -ge 500 ]; then
  printf 'GPU1 is not free: %s MiB; clone was not launched.\n' "$used" > clone_queue.log
  printf '1\n' > clone_queue.exit
  exit 1
fi
screen -dmS sttrack_m55_clone_train_v2_20260906 bash "$root/run_training.sh" clone 1
status=$?
date --iso-8601=seconds > clone_queue_dispatched_at.txt
printf '%s\n' "$status" > clone_queue.exit
exit "$status"
