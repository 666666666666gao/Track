#!/usr/bin/env bash
set -eu
root=/root/autodl-tmp/sttrack_m54_template_reader_v1_20260906
native=/root/autodl-tmp/sttrack_default_rgbd_ope_v1_20260906
echo $$ > "$root/queue.pid"
trap 'printf "%s\n" "$?" > "$root/queue.exit"' EXIT
first_check=$(date -u -d '2026-09-05 19:30:00' +%s)
delay=$((first_check - $(date -u +%s)))
if [ "$delay" -gt 0 ]; then
  printf 'Waiting until 2026-09-05 19:30:00 UTC before checking native tracking exits\n'
  sleep "$delay"
fi
while [ ! -f "$native/tracking_depthtrack.exit" ] || [ ! -f "$native/tracking_cdtb.exit" ]; do
  date -u '+%Y-%m-%dT%H:%M:%SZ native tracking still running; next check in 240 seconds'
  sleep 240
done
[ "$(cat "$native/tracking_depthtrack.exit")" = 0 ]
[ "$(cat "$native/tracking_cdtb.exit")" = 0 ]
[ "$(cat "$root/contract.exit")" = 0 ]
[ "$(cat "$root/runtime_contract.exit")" = 0 ]
[ ! -e "$root/controller.pid" ]
date -u '+%Y-%m-%dT%H:%M:%SZ starting frozen M54 collection-training-recursion pipeline'
bash "$root/run.sh"
