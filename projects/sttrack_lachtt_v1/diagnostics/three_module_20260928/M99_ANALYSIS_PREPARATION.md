# M99 completed-result analysis preparation

The M99 GPU sanity and final visual control are still pending M98's terminal
input audit. This artifact is a CPU result-analysis tool, not a completed M99
result, new checkpoint, training invocation or scientific qualification.

`analyze_ab_visual_control.py` accepts the completed M99 result and its actual
final weight file. It checks the registered seed, 12 epochs/480 steps, Empty
input and capacity boundaries; hashes the supplied final weights; recomputes
every development stratum and all four predeclared checks from the 495 unique
event records; and exports all 22 sequences, all events, the descriptive
M91/M96 comparison, JSON and Markdown. Failed scientific checks remain visible.
The analyzer has no training, remote invocation, tracker-state commit or
promotion operation. It does not independently rerun inference or validate
the tensor inputs; those remain separate runtime/receipt checks.

Actual preparation verification used the 495 saved events in each of M91 and
M96, 990 historical records in total. Their recomputed overall counts, means,
rescues and breaks agree with the archived results. This verifies the
arithmetic on real historical rows; it does not verify M99's future result or
the complete M99 command path. The receipt is
`m99_prepared/analysis_arithmetic_check.json`. The fresh same-family/provisional
source review passed after correction of a terminal-print keyword placement
and precise hash-verification wording. See `M99_ANALYSIS_REVIEW.md`; the actual
completed-M99 command and weight verification remain pending.

After downloading the actual final run result and weight to `m99_completed/`:

```powershell
uv run --no-project python projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/analyze_ab_visual_control.py --result projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/m99_completed/final_train/result.json --weights projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/m99_completed/final_train/final.pt --output projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/m99_completed/analysis
```

The historical comparison differs in representation, reference origin and
trainable capacity. It is descriptive, not a matched causal ablation. The
former development22 is reused development within DepthTrack Train; the
fixed-state metrics are not official DepthTrack Test, CDTB or VOT results.
