#!/usr/bin/env bash
set -euo pipefail
repo=/root/autodl-tmp/rgbd_baselines/STTrack_lachtt_v1
run=/root/autodl-tmp/sttrack_m42_local_spatial_v1_20260905
py=/root/autodl-tmp/envs/sttrack/bin/python
cd "$repo"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
"$py" tools/test_sttrack_m42.py > "$run/test.log" 2>&1
"$py" - <<'PY'
import hashlib, json, pathlib, time
r=pathlib.Path('/root/autodl-tmp/sttrack_m42_local_spatial_v1_20260905')
files=['tools/run_sttrack_m42.sh','tools/train_sttrack_m42.py','tools/test_sttrack_m42.py']
hashes={f:hashlib.sha256(pathlib.Path(f).read_bytes()).hexdigest() for f in files}
smoke=json.loads((r/'smoke_receipt.json').read_text())
assert smoke['status']=='complete' and smoke['events']==6
assert all(x['max_bbox_error_px']==0 and x['max_score_error']==0 for x in smoke['sequences'])
assert not list((r/'features').glob('*.pt'))
assert not (r/'training_result.json').exists()
(r/'launch.json').write_text(json.dumps(dict(source_sha256=hashes,started_unix=time.time(),
    spec_sha256=hashlib.sha256((r/'spec.json').read_bytes()).hexdigest(),
    smoke_receipt_sha256=hashlib.sha256((r/'smoke_receipt.json').read_bytes()).hexdigest()),indent=2)+'\n')
PY
CUDA_VISIBLE_DEVICES=0 "$py" tools/collect_sttrack_m42.py --root "$run" --shard 0 > "$run/shard0.log" 2>&1 &
first=$!
CUDA_VISIBLE_DEVICES=1 "$py" tools/collect_sttrack_m42.py --root "$run" --shard 1 > "$run/shard1.log" 2>&1 &
second=$!
printf '%s\n%s\n' "$first" "$second" > "$run/collector_pids.txt"
set +e
wait "$first"; first_status=$?
wait "$second"; second_status=$?
set -e
echo "$first_status" > "$run/shard0.exit"
echo "$second_status" > "$run/shard1.exit"
if [[ "$first_status" -ne 0 || "$second_status" -ne 0 ]]; then
    exit 1
fi
CUDA_VISIBLE_DEVICES=0 "$py" tools/train_sttrack_m42.py --root "$run" > "$run/training.log" 2>&1
