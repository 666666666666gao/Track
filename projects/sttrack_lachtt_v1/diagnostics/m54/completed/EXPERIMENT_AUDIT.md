Completed the bounded read-only M54 full-recursive Type-A advisory review. I did not use SSH, GPU, inference, tuning, or project/raw edits. I read the request manifest and primary artifacts directly, verified local hashes, and independently recomputed the full 22-sequence default-vs-reader metrics from the downloaded raw boxes and copied dataset GT. I did not import the executor metric function; I only read it for line evidence.

Overall verdict: **WARN**, with **no metric-arithmetic defect found**. The completed recursive metrics and per-sequence table are internally supported by the downloaded raw bundle, but the proposed reader **does not pass its own promotion gate** because it creates a new H10 failure on `mobilephone02_indoor`. Scope remains limited to the already-used DepthTrack Train development split; there is no public DepthTrack Test/CDTB/VOT/full127 result here.

| Check | Status | Finding |
|---|---:|---|
| A. Ground-truth provenance | PASS | Metrics use copied original DepthTrack Train dataset GT after recursive outputs were sealed. This is real GT for the development split, not model-derived GT. |
| B. Score normalization | PASS | Metrics are continuous xywh IoU, summed over valid GT frames and divided by valid-frame counts or macro-averaged by sequence. No self-normalization by prediction statistics found. |
| C. Result existence / hashes | PASS with limit | The 75 downloaded files exist and hash-match the local download binding; all three terminal exits are `0`; result/log summary agrees. Local bundle does not include `reader_final.pth` or `training_result.json`, so I cannot independently rehash those bytes in this review. |
| D. Metric execution / dead code | PASS | The runner’s analyze path is wired: it verifies recursive/default outputs, loads GT after output verification, computes per-arm metrics, gates, and writes `recursive_result.json`. |
| E. Scope and claim strength | WARN | Evidence is full 22-sequence DepthTrack Train development recursion, not public benchmark evaluation. The reader improves aggregate mean IoU and reduces low/H10 counts, but primary promotion is false because successful-sequence protection fails. |
| F. Evaluation type | PASS / real_gt | Recursive metrics are `real_gt` over dataset GT. The phone diagnostic is a post-hoc descriptive real-GT chronology where GT is valid, with explicit non-causal limits. |

Independent recomputation matched the published metrics. Differences against `recursive_result.json`:

| Quantity | Published | Independent | Difference |
|---|---:|---:|---:|
| default valid frames | 28,897 | 28,897 | 0 |
| default IoU sum | 18,847.382327003917 | 18,847.382327003920 | 3.64e-12 |
| default mean IoU | 0.6522262631762438 | 0.6522262631762440 | <1e-15 |
| default macro sequence mean IoU | 0.6843364163600396 | 0.6843364163600397 | <1e-15 |
| default low-IoU frames | 7,397 | 7,397 | 0 |
| default H10 episodes | 75 | 75 | 0 |
| reader valid frames | 28,897 | 28,897 | 0 |
| reader IoU sum | 19,440.002302580020 | 19,440.002302580022 | 3.64e-12 |
| reader mean IoU | 0.6727342735432750 | 0.6727342735432752 | <1e-15 |
| reader macro sequence mean IoU | 0.7102239826379665 | 0.7102239826379665 | 0 |
| reader low-IoU frames | 6,692 | 6,692 | 0 |
| reader H10 episodes | 65 | 65 | 0 |

These aggregate values are published in `recursive_result.json:7-23` and repeated in `analysis.log:7-23`. The report’s table gives the same rounded values in `RESULT_REPORT.md:7-13`. The two nonzero IoU-sum differences are floating-point summation-order noise only; all counts and all CSV values matched.

All 22 rows in `per_sequence.csv` matched my recomputation exactly at the written precision:

| Sequence | Valid | Default mean | Reader mean | Gain pp | Default low | Reader low | Default H10 | Reader H10 | Reads |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bag05_indoor | 587 | 0.827925317240346 | 0.850733678353793 | +2.280836111345 | 25 | 10 | 1 | 0 | 15 |
| ball07_indoor | 2,179 | 0.365679729312190 | 0.359207303389645 | -0.647242592254 | 1,288 | 1,314 | 8 | 10 | 23 |
| ball19_indoor | 1,380 | 0.837602338699389 | 0.838543023718474 | +0.094068501908 | 19 | 21 | 1 | 1 | 1 |
| beautifullight01_indoor | 910 | 0.614816781458293 | 0.623494931975539 | +0.867815051725 | 270 | 261 | 4 | 4 | 18 |
| book06_indoor | 1,545 | 0.656160005287550 | 0.787821829442292 | +13.166182415474 | 381 | 145 | 3 | 3 | 89 |
| car01_indoor | 1,402 | 0.390289898876982 | 0.390268875522595 | -0.002102335439 | 814 | 814 | 3 | 3 | 17 |
| car02_indoor | 565 | 0.693114717410216 | 0.692022115122429 | -0.109260228779 | 143 | 143 | 2 | 2 | 2 |
| colacan01_indoor | 2,801 | 0.101155278999350 | 0.264635533705365 | +16.348025470601 | 2,464 | 1,921 | 10 | 10 | 10 |
| colacan04_indoor | 567 | 0.276557505199776 | 0.621114160469282 | +34.455665526951 | 392 | 177 | 12 | 6 | 79 |
| container01_indoor | 621 | 0.925852435378341 | 0.925403025601636 | -0.044940977671 | 4 | 3 | 0 | 0 | 13 |
| cup08_indoor | 1,321 | 0.831394862945442 | 0.841161933548989 | +0.976707060355 | 103 | 86 | 1 | 1 | 2 |
| cup10_indoor | 2,147 | 0.866296953309497 | 0.851510682156156 | -1.478627115334 | 77 | 107 | 1 | 2 | 28 |
| cup13_indoor | 1,658 | 0.770102585800097 | 0.769692825554475 | -0.040976024562 | 111 | 112 | 3 | 3 | 31 |
| egg_indoor | 3,433 | 0.734777786594881 | 0.736101380937678 | +0.132359434280 | 523 | 522 | 9 | 8 | 63 |
| flower02_wild | 467 | 0.650547406320178 | 0.786661736297665 | +13.611432997749 | 100 | 24 | 2 | 0 | 74 |
| flower03_indoor | 547 | 0.838305570152511 | 0.837165688043937 | -0.113988210857 | 0 | 0 | 0 | 0 | 3 |
| flowerbasket_indoor | 851 | 0.762025010490677 | 0.786390415820616 | +2.436540532994 | 123 | 100 | 3 | 1 | 39 |
| ghostmask_indoor | 1,025 | 0.716086083190501 | 0.853806861102258 | +13.772077791176 | 193 | 16 | 3 | 0 | 42 |
| glass03_indoor | 570 | 0.757593918071359 | 0.894070840446714 | +13.647692237536 | 111 | 30 | 3 | 1 | 6 |
| mobilephone01_indoor | 1,217 | 0.729845150717263 | 0.561510779508554 | -16.833437120871 | 207 | 435 | 5 | 6 | 4 |
| mobilephone02_indoor | 610 | 0.851274441111706 | 0.568574590447532 | -28.269985066417 | 1 | 205 | 0 | 1 | 4 |
| notebook02_indoor | 2,494 | 0.857997383354329 | 0.785035406869636 | -7.296197648469 | 48 | 246 | 1 | 3 | 122 |

The CSV rows are at `per_sequence.csv:2-23`. The critical failing row is `mobilephone02_indoor`, where default H10 is `0` and reader H10 is `1` at `per_sequence.csv:22`; `recursive_result.json:365-371` gives the same reader metrics.

Promotion gate recomputation:

| Gate | Rule | Independent result | Published |
|---|---|---:|---:|
| mean_iou | reader mean ≥ default mean + 0.01 | PASS: gain = 0.0205080103670312 | `true` |
| fewer_low_frames | reader low frames < default low frames | PASS: 6,692 < 7,397 | `true` |
| no_episode_increase | reader H10 ≤ default H10 | PASS: 65 ≤ 75 | `true` |
| sequence_coverage | positive sequences ≥ 3 | PASS: 12 ≥ 3 | `true` |
| successful_sequence_protection | no sequence with default H10=0 and reader H10>0 | FAIL: `mobilephone02_indoor` | `false` |
| primary_pass | all gates true | FAIL | `false` |

The predeclared gate is in `spec.json:26-32` and in the plan at `EXPERIMENT_PLAN.md:29-31`. The published gate values are in `recursive_result.json:383-397` and `analysis.log:25-39`. The report correctly states `primary_pass=false` and no public evaluation/continuation at `RESULT_REPORT.md:15` and `RESULT_REPORT.md:33`.

For valid/invalid denominators, I recomputed 33,130 total frames across the 22 reader trajectories, excluded the 22 initialization frames, counted 4,211 invalid non-initial GT frames, and got 28,897 valid evaluation frames: `33,130 - 22 - 4,211 = 28,897`. That matches `review_inputs/binding.json:212-216`, `recursive_result.json:9-23`, and the report table at `RESULT_REPORT.md:7-11`. The metric contract in the reused analyzer also marks GT invalid when nonfinite or non-positive-size and excludes frame 0 at `analyze_sttrack_m42_recursive.py:20-30`.

One minor wording issue: the analyzer source defines low frames as `IoU <= .1` at `analyze_sttrack_m42_recursive.py:28`, while the report table labels the row as `IoU<0.1` at `RESULT_REPORT.md:10`. I found zero frames exactly at 0.1 in either arm, so this does not change any reported count. The report label should be tightened to `IoU ≤ 0.1` for exactness.

Hash, receipt, and execution evidence:

- `recursive.exit`, `analysis.exit`, and `controller.exit` are all `0` (`recursive.exit:1`, `analysis.exit:1`, `controller.exit:1`).
- The recursive receipt lists all 22 sequences with frame counts, initial-read counts, and trajectory hashes from `recursive_receipt.json:3-158`, and records `source_unchanged: true` plus `subsequent_gt_opened: false` at `recursive_receipt.json:160-163`.
- `recursive.log:2-23` contains the same 22 per-sequence receipt objects and hashes.
- `download_binding.json:4-24`, `download_binding.json:105-123`, and `download_binding.json:301-307` bind the downloaded exits/logs/result/receipt/spec and raw review files; my local rehash over all 75 entries found no mismatches.
- `review_inputs/binding.json:212-216` records 22 sequences, 33,130 frames, all recursive outputs sealed before GT copy, no metrics computed during export, and native/default boxes copied from historical trace values while GT was byte-copied from the DepthTrack Train dataset.
- `export_m54_recursive_review.py:11-23` checks all exits and recursive receipt hashes before export; `export_m54_recursive_review.py:25-35` copies default/native `frame_index` and `public_bbox` from bound historical traces; `export_m54_recursive_review.py:37-52` copies dataset GT only after prediction-family verification; `export_m54_recursive_review.py:56-62` writes the binding with `metrics_computed=False`.
- `download_m54_recursive_completion.py:19-30` checks remote exits and expected hashes before download; `download_m54_recursive_completion.py:33-42` hashes every downloaded file; `download_m54_recursive_completion.py:45-50` verifies result/receipt linkage and labels the bundle as raw review data.

Source and binding checks:

- The downloaded `spec.json` hash is `1ca1387e6eb33c897e12d5e0c10b746d48257e7745906c431e8ac857f63d7267`, matching `recursive_result.json:3`, `analysis.log:3`, and `spec.json:34`.
- The M44 parent spec hash in `spec.json:4` matches the current publisher `diagnostics/m44/spec.json` hash; the M44 baseline trace hashes are declared at `diagnostics/m44/spec.json:21-24`.
- The M54 spec declares frozen native STTrack plus trained two-view RGB-D template reader and no language at `spec.json:15-16`.
- The M54 source hashes and `run.sh` hash declared at `spec.json:35-45` matched the current publisher files in my local hash pass.
- The runner checks training completion and checkpoint hash before recursion at `run_sttrack_m54.py:20-24`, loads the saved reader state strictly at `run_sttrack_m54.py:91-95`, writes frame 0 from the initialization bbox and subsequent frame outputs at `run_sttrack_m54.py:107-113`, and records initial-read counts plus trajectory hashes at `run_sttrack_m54.py:114-122`.
- The runner’s analysis path verifies recursive receipt status, sequence coverage, per-file trajectory hashes, frame order, positive finite boxes, and legal choices before loading GT at `run_sttrack_m54.py:28-62`. It computes aggregate metrics and gates at `run_sttrack_m54.py:63-82`.

The local review bundle does **not** include `reader_final.pth`, `training_result.json`, or `training.exit`. Therefore I can verify the recursion result’s declared checkpoint/training hashes and the runner source’s execution-time assertions, but I cannot independently rehash the trained reader checkpoint or training result bytes from this particular downloaded bundle. This is an evidence limit, not a metric mismatch in the completed recursive review.

Phone chronology verification:

- `phone_trajectory_diagnostic.json:2-38` reports `mobilephone02_indoor` initial reads at frames 414, 568, 649, and 691, first bbox difference at 414, and a reader H10 run from frame 497 to 701 with length 204.
- I independently reproduced first difference frame 414 and the H10 run `(497, 701, 204)`.
- The GT at frame 414 is invalid (`groundtruth/mobilephone02_indoor.txt:415` is `nan,nan,nan,nan`), so the first read cannot itself produce a valid-frame IoU drop. Later reads at frames 568, 649, and 691 have valid GT (`groundtruth/mobilephone02_indoor.txt:569`, `:650`, `:692`) and occur after the low-overlap run has already started.
- `phone_trajectory_diagnostic.json:40-101` reports `mobilephone01_indoor` initial reads at frames 746, 749, 760, and 1351, first bbox difference at 746, and six reader H10 runs. I independently reproduced those read frames, first difference, and all six H10 run boundaries.
- The four listed `mobilephone01_indoor` read frames have invalid GT: `groundtruth/mobilephone01_indoor.txt:747`, `:750`, `:761`, and `:1352`.
- The diagnostic correctly limits itself: both phone entries say the sealed native and reader trajectories after first divergence are different recursive states and provide no same-state counterfactual or attribution to a specific update (`phone_trajectory_diagnostic.json:38`, `phone_trajectory_diagnostic.json:101`). The report repeats that limitation at `RESULT_REPORT.md:25`.

Supported claims:

- Supported: completed M54 reader recursion exists for all 22 DepthTrack Train development sequences, with 33,130 total frames and 28,897 valid non-initial GT frames.
- Supported: aggregate valid-frame mean IoU improves from `0.6522262631762438` to `0.672734273543275`, a gain of `+2.05080103670312` percentage points.
- Supported: low-IoU frames decrease from `7,397` to `6,692`; H10 episodes decrease from `75` to `65`; 12 sequences improve in mean IoU.
- Supported: all published `per_sequence.csv` rows match independent recomputation from the downloaded raw default boxes, reader boxes, and copied GT.
- Supported: the reader fails the predeclared primary promotion gate because `mobilephone02_indoor` goes from default H10 `0` to reader H10 `1`.
- Supported: the phone diagnostic is descriptive chronology over sealed divergent trajectories, not causal attribution.

Qualified claims:

- Qualified: “real GT” means copied DepthTrack Train dataset GT for the development split, not public benchmark GT and not an untouched holdout for this project.
- Qualified: default/native boxes in the review bundle are copied from bound historical traces; I verified the copied default JSONs and their binding, but I did not independently download or inspect the full original source shard files in this review.
- Qualified: checkpoint/source binding is supported by the spec, receipts, source hashes, and runner assertions; local independent checkpoint-byte verification is unavailable because the trained weights and training result are not included in the 75-file review bundle.

Unsupported claims:

- Unsupported: that M54 is ready for public automatic launch or promotion. `primary_pass` is false.
- Unsupported: that this proves DepthTrack Test, CDTB, VOT, M39 EAO/ACC/ROB, or full127 performance. The result explicitly sets `public_evaluation=false` at `recursive_result.json:395`, and the scope line is Train development only at `recursive_result.json:396`.
- Unsupported: that a fitting-window/static rescue count is equivalent to trajectory rescue. The report itself warns against treating static recovery counts as trajectory recoveries at `RESULT_REPORT.md:33` and frames the prior TSG finding as separate at `RESULT_REPORT.md:37`.
- Unsupported: that any individual phone read caused the later failure without same-state counterfactual replay. The phone diagnostic explicitly rejects that attribution at `phone_trajectory_diagnostic.json:38` and `phone_trajectory_diagnostic.json:101`.

Concrete action items:

1. Keep `primary_pass=false` and stop/freeze this reader as the current report says. The successful-sequence protection gate failed on `mobilephone02_indoor`.
2. In the report/table wording, change the low-frame label from `IoU < 0.1` to `IoU ≤ 0.1`, matching the actual analyzer threshold. This is a wording correction only; counts are unchanged because no frame had IoU exactly 0.1.
3. If a later review needs full base/head binding, include `reader_final.pth`, `training_result.json`, and `training.exit` in the downloaded review bundle so the trained checkpoint and training-result hash can be rechecked locally.
4. Preserve the scope wording: “DepthTrack Train development recursive diagnostic,” not public benchmark performance and not causal attribution for individual reads.