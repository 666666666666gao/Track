# M82 Full152 VOT failure candidate diagnostic

Prepared after the factor4 crop census showed 122 of 124 newly failed anchors
still contained the GT center at the first confirmed low-overlap segment start.
This is a posthoc external diagnostic, not training data or a new tracker score.

The 124 anchors are fixed by M82 failure and native success, before observing
candidate maps. Each anchor replays Category from its original VOT initialization
through failure-start + 9. The ten preceding and ten following positions save
all 256 score/size/offset outputs from Category, same-weight Empty and the
unadapted head. Only Category commits bbox, query or templates. The native head
inherits the Category history; it is not an independent native tracker.

The CPU preparation is complete: 52,262 tracking calls, 2,480 readout positions,
balanced into 26,138 and 26,124 calls. No GPU preflight or full replay has run yet.
Original VOT initialization uses the already audited TraX float32/four-decimal
conversion. Every replayed prediction must match the sealed trajectory after
that serialization, and confidence must match at float32 precision.

Run only after the current eight OPE content controls finish. Use the existing
training environment, without installing packages or editing its runtime:

```bash
root=/root/autodl-tmp/sttrack_full152_vot_readout_20260926
model_python=/root/autodl-tmp/envs/sttrack/bin/python
metric_python=/root/miniconda3/envs/mplt/bin/python
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
CUDA_VISIBLE_DEVICES=0 "$model_python" -u "$root/readout.py" --shard 0 --preflight
```

Inspect successful bbox/score parity and finite dense maps before running both
shards on GPU0/GPU1 with `--shard 0` / `--shard 1`, without `--preflight`.
After both receipts are complete, run `"$metric_python" "$root/analyze.py"`.
Do not restart or overwrite a failed output directory; investigate its receipt.

The analysis reconstructs dense boxes using the M85 decoding formula, checks
selected locations against direct Torch decoding, and uses the existing VOT
bounded raster overlap with `_ignore`. It distinguishes missing candidates,
raw/Hann selection, same-state Empty and geometry/score paths. Best candidate
overlap uses GT only after readouts seal and is an upper bound, not a deployment
selection rule. Pre-onset frames belong to selected failed anchors; they do not
establish the false-intervention rate on independent healthy trajectories.

The preparation and readout source hashes are frozen in `spec.json`; analysis
source is recorded in its result. GPU duration will be estimated from preflight.
This diagnostic does not change either Full152 final checkpoint or the ongoing
content-control queue.

The CPU controller is queued as PID 808026 (`queue_launch.json`). It sleeps until
2026-09-26 16:50 CST, then checks the existing content controller's completion
every 300 seconds. It requires all eight complete metric files and both GPUs
free, runs one-anchor preflight, and only on success starts both replay shards.
Errors stop this queue; no automatic restarts or changed tolerances are used.
`logs/queued_controller.log`, step exit files and `replay_launch.json` expose
actual execution. The current state is preparation/CPU wait, not GPU completion.
