#!/bin/bash
set -u
root=/root/autodl-tmp/sttrack_m73_paired_lexical_replication_20260907
python=/root/autodl-tmp/envs/sttrack/bin/python
cd "$root" || exit 1
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
"$python" -u /root/autodl-tmp/m73_paired_lexical_replication_20260907.py eligible > eligibility.log 2>&1
status=$?; printf '%s\n' "$status" > eligibility.exit
if [ "$status" -ne 0 ]; then printf '1\n' > controller.exit; exit 1; fi
for seed in 2027 2028; do
    printf '%s\n' "$seed" > current_seed.txt
    bash "seed$seed/run_pair.sh" > "seed$seed/queue.log" 2>&1
    status=$?; printf '%s\n' "$status" > "seed$seed.exit"
    if [ "$status" -ne 0 ]; then printf '1\n' > controller.exit; exit 1; fi
done
"$python" -u /root/autodl-tmp/m73_paired_lexical_replication_20260907.py summarize > analysis.log 2>&1
status=$?; printf '%s\n' "$status" > analysis.exit
printf '%s\n' "$status" > controller.exit
exit "$status"
