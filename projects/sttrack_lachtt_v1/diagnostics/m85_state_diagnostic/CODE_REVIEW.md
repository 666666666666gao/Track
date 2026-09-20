# M85 code review

**PASS: no remaining blocking or non-blocking implementation findings.** This is a fresh Codex review with `review_independence: same-family` and `acceptance_status: provisional`. The parent confirmed the actual reviewer invocation: `gpt-6-astra`, reasoning effort `max`, `fork_turns: none`.

This review covers the final local bytes of the plan, spec, replay, queue, posthoc analyzer and launcher, compared with the supplied M83 driver and relevant frozen M84 code. It is a code and syntax review. The reviewer did not use SSH, run CUDA, load dataset GT, run the analyzer, create a model, or add seeds. The reviewer changed no implementation files.

The final implementation correctly retains Category head outputs before alternate hook calls overwrite capture; the alternate branches use the captured RGB/depth/fused/initial state without entering the tracker state-update path. Exact Category bbox/score replay and Empty/native map equality are enforced by assertions. The finite dense map layout, crop geometry, output-window storage and default template-write eligibility agree with the frozen M84 implementation. The queue prevents the full replay after any failed preflight.

The final analyzer checks the complete 2616-position receipt and all three prediction/dense artifact hashes before reading subsequent GT. Its dense decoder follows the frozen center head and clipping conventions, checks each selected raw/Hann box within 1e-4 pixel, and uses saved selected boxes for primary IoU. The Category-low count partition is exhaustive; alternate-correct and all-heads-no-correct capacities are separate counts. H10 intervals are half-open `[start,end)`, require at least 10 valid frames at Hann IoU <= 0.1, and stop at invalid GT.

Earlier review observations are resolved in these final bytes: direct native-weight and integration digest checks are now present (`same_state.py:33-34`); the analyzer now records continuous low-overlap intervals (`analyze_saved.py:29-37,93`) and alternate/all-head capacity (`analyze_saved.py:90-91`).

Local verification passed: Python 3.12.6 syntax compilation for `same_state.py`, `analyze_saved.py`, and `launch.py`; `bash -n` for `run_replay.sh`; JSON parsing; plan/replay/analyzer/queue source bindings; the prior M83 driver digest; and the fixed 2616 tracking-call total. No CUDA execution or numerical test is implied by these checks.

PASS permits the defined CUDA preflight. Actual remote hashes, bitwise replay, Empty/native equality, output archives and full-run completion still require real receipts. All alternate results share the Category history. Candidate existence remains hindsight capacity, and candidate absence is bounded to the tested heads; neither proves trajectory repair or identifies an earlier state update as the cause. These posthoc selected cases do not support formal metrics, unbiased prevalence, or M84 promotion.

| Check | Evidence | Result |
|---|---|---|
| chronological exact replay | `same_state.py:78-91`, `same_state.py:130` | Static PASS |
| capture and hook overwrite | `same_state.py:56-57`, `same_state.py:92-101`, `M84/code/lib/models/sttrack/sttrack.py:147-154` | Static PASS |
| alternative state isolation | `same_state.py:95-110`, `M84/code/lib/models/sttrack/centered_semantic_adapter.py:12-20`, `M84/code/lib/models/sttrack/semantic_spatial_adapter.py:25-52`, `M84/code/lib/models/sttrack/sttrack.py:175-205` | Static PASS |
| Empty/native and dense layout | `same_state.py:102-107`, `same_state.py:137-139`, `M84/code/lib/models/layers/head.py:142-156` | Static PASS |
| crop and template eligibility | `same_state.py:111-133`, `M84/code/lib/train/data/processing_utils.py:32-41`, `M84/code/lib/test/tracker/sttrack.py:119-139`, `M84/code/lib/test/tracker/sttrack.py:192-198` | Static PASS |
| source and checkpoint binding | `same_state.py:29-40`, `same_state.py:51-54`, `same_state.py:80-81`, `launch.py:8-11` | Static PASS |
| preflight and launch gate | `run_replay.sh:5-13`, `launch.py:8-21` | Static PASS |
| GT only after sealed predictions | `same_state.py:67-91`, `analyze_saved.py:40-54` | Static PASS |
| dense decoder and IoU | `analyze_saved.py:11-27`, `analyze_saved.py:67-77`, `M84/code/lib/models/layers/head.py:142-156`, `M84/code/lib/utils/box_ops.py:97-106` | Static PASS |
| valid frames and count partition | `analyze_saved.py:62-66`, `analyze_saved.py:79-93` | Static PASS |
| low-overlap intervals | `analyze_saved.py:29-37`, `analyze_saved.py:93` | Static PASS |

Full details and review scope for each frozen file are in `code_review.json`.

| Reviewed file | SHA-256 |
|---|---|
| `EXPERIMENT_PLAN.md` | `93c66f1943c275055741028b7db92d5f5a9ac18ce4cdf2486d2028122b2b2ebd` |
| `spec.json` | `748b11527aa7302490a37cfdf9b2b7643239e604943880e675d2e480b4e533ec` |
| `same_state.py` | `6959a6e632bccea2469bae523d924698cbc6dfd32b08b38ec0e1bf1658326f41` |
| `run_replay.sh` | `74a1d147b310d76df45eeb3ce2fbb269d388e86ec1673d33826249db2c46edcb` |
| `analyze_saved.py` | `6bc7f5cbf5a451d19f3b03cc7b1d65d51f1fe17dd25f460c1fe602ea1d4265d0` |
| `launch.py` | `0663abe161ed7d83fc409543cdc966dacfcdd1852f043f44c56bcb92927f9106` |
| `M83/m83_same_state.py` | `976ff0c134f9f839db4e361344446632a5377ae41b8ac9cfa257ac99c00a589c` |
| `M84/code/lib/models/sttrack/centered_semantic_adapter.py` | `59be315803924f429d3e431b753c4cc7a6acc98a97a02cb8e370f7078fbe36a5` |
| `M84/code/lib/models/sttrack/semantic_spatial_adapter.py` | `dde93520ed569ede7f4f0b9a0c10d0b6ff179135b9928b2b3e03f41dfd3f4d88` |
| `M84/code/lib/test/tracker/sttrack_semantic.py` | `9283fc1c54f1df157eb9d28fec69b49c8655e2898a01342228452809fdd77922` |
| `M84/code/lib/test/tracker/sttrack.py` | `5b114e384942ecf2740e1b2da93bea855eb38246786c62e49bf5ce70b78ff8c6` |
| `M84/code/lib/models/sttrack/sttrack.py` | `aa75dfcb6b8990a10293b7415332f6fe293e8584d0ad7348423e3a4a48129057` |
| `M84/code/lib/models/layers/head.py` | `62cb2f363ea8bce491c9e6fc528076dd1e14c7d371ad69d6efd0b4062d063326` |
| `M84/code/lib/train/data/processing_utils.py` | `7aca916f5e5f62e1865fbd322ffb63dc08197c544a241938b6c822d4bfb89bf6` |
| `M84/code/experiments/sttrack/deep_rgbd_256_lachtt_v1.yaml` | `b6bda3238c9dd001aab62d87234d9ccecf1ae1cee3bd7bea5f21d6368ff4b344` |
| `M84/code/lib/train/dataset/depth_utils.py` | `f97336af94a3ec196ad239603896fc0a2387dd1d61b48a4fbe2e055d9466e92e` |
| `M84/code/lib/test/tracker/sttrack_initial_instance_observation.py` | `2f63f49f90b08f3e205c9bb48cb22aad1d800b6bc7d1d823746c638bf9ffddde` |
| `M84/code/lib/test/tracker/data_utils.py` | `873337d0419ecfdd08b64123cdc22a4936ccdd6af98a092bc2106c32cf401fe5` |
| `M84/code/lib/utils/box_ops.py` | `cee9d309a9c4a6c5f00fab575719bc7bf8d865127c91b10d4fb233996a111fbc` |
