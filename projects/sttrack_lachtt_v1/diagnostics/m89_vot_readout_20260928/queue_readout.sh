#!/usr/bin/env bash
set -euo pipefail

content=/root/autodl-tmp/sttrack_m89_content_20260927
root=/root/autodl-tmp/sttrack_m89_vot_readout_20260928
while [ ! -f "$content/complete.exit" ]; do
    sleep 300
done
exec bash "$root/run_readout.sh"
