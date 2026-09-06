Same-GPT-family Type-A advisory review only. I inspected the v2 request manifest and the listed local artifacts; I did not edit files, contact the server, use GPU, rerun training, or infer current remote completion. I also read the previous v1 trainer only to verify the requested v1/v2 trainer delta.

Roots used below:

- `E = C:\Users\gb\.codex_remote_staging\m55_training_evidence_20260906`
- `P = C:\Users\gb\.codex_track_publish_m29_20260902\projects\sttrack_lachtt_v1`
- `D = C:\Users\gb\.codex_track_publish_m29_20260902\docs`

Overall verdict: **PASS for v2 dataset-contract readiness and local evidence binding; WARN for evidence maturity.** I found no concrete blocker in the v2 dataset-contract/training-entry artifacts. The remaining limitation is that the retained training and recursive/public metrics are still not completed evidence in the reviewed local artifacts.

| Check | Status | Finding |
|---|---:|---|
| Failed partial training vs completed stages | PASS | v1 is correctly represented as an aborted partial run, not a completed training result. |
| Data exception / GT provenance | PASS | The `toy07_indoor_320` exception is narrow, hash-bound, and already documented in existing source/project evidence. |
| v1/v2 trainer delta | PASS | The v2 trainer adds only the `toy07` tail contract and explicit frame-file checks; sampling, seeds, optimizer, loss, budget, and gates remain fixed by spec. |
| Contracts and local byte binding | PASS | v2 dataset contract and both sampler contracts exist and bind to the v2 spec/trainer hashes; my local hash pass matched all 40 entries in `revision_download_binding.json`. |
| Scope / claims | WARN | These are contract/preflight/launch artifacts. They do not establish retained training completion, final weights, recursive metrics, or public benchmark performance. |

Evidence and details:

**1. The original v1 failure is real, partial, and correctly scoped.** `E\training_abort.json:2-8` records `aborted_data_contract_omission`, 20 optimizer steps, 80 optimized microbatches, `training_complete=false`, no saved trained weight, and clone training not started. The failure stack is concrete: `E\train_control.log:3-5` reports inability to read `/toy07_indoor_320/color/00001406.jpg`, and `E\train_control.log:25-35` shows the DataLoader path ending in OpenCV `cvtColor` failure on an empty source image. `E\clone_queue_cancelled.json:2-5` records that the paired clone arm was paused before starting because control failed on that unreadable frame. The binding also records nonzero v1 exits via `E\revision_download_binding.json:34-40`.

The apparent `old_completed_batches=82` in `E\v2\dataset_contract.json:5` should be read as old logged batch rows, not optimizer-completed microbatches. My local read of the bound old logs found 82 `batches.jsonl` rows but only 20 `steps.jsonl` rows, matching `E\training_abort.json:4-5`. Avoid wording that says v1 completed 82 optimization batches.

**2. The data exception is narrow and supported by existing evidence.** The inventory shows `toy07_indoor_320` has 1,406 GT rows but only 1,367 RGB and 1,367 depth frames, with min/max frame range 1..1367: `E\dataset_inventory.json:1067-1078`. The missing RGB tail starts at 1368 and reaches 1406 (`E\dataset_inventory.json:1079-1118`), and the missing depth tail likewise spans 1368..1406 (`E\dataset_inventory.json:1120-1159`); there are no extras for that sequence (`E\dataset_inventory.json:1160-1162`). The file-level totals are 39 missing RGB and 39 missing depth entries: `E\dataset_inventory.json:1219-1220`. My local parse found this is the only fit-sequence GT/frame mismatch.

This is not an invented new truncation rule. The existing analyzer already special-cases only `toy07_indoor_320`, requiring `len(lines)==1406`, `expected_frames==1367`, and the exact GT SHA before using `lines[:expected_frames]`: `P\overlay\tools\analyze_sttrack_lachtt_train152_gatea.py:106-117`. The master document records the same historical contract: strict mismatch `RGB=1367, Depth=1367, GT=1406` at `D\RGBD_LANGUAGE_TRACKING_PROJECT_MASTER.md:10755-10760`, the exact full-GT SHA and 39 annotation-only tail at `D\RGBD_LANGUAGE_TRACKING_PROJECT_MASTER.md:10762`, and that this is a fail-closed exact tail contract rather than generic truncation at `D\RGBD_LANGUAGE_TRACKING_PROJECT_MASTER.md:10764`.

**3. V2 reconnects that existing contract without adding a broad fallback.** `E\v2\prepare_revision.py:9-20` derives counts from the existing inventory, asserts RGB/depth count equality, asserts no extras, checks normal sequences as `gt_rows==count`, and for `toy07_indoor_320` alone requires `count==1367`, `gt_rows==1406`, missing lists exactly `1368..1406`, and the exact full-GT SHA. It then asserts the count set equals the existing fit split. The v2 spec binds the previous spec, abort record, and inventory hashes at `E\v2\training_spec.json:250-253`, lists 63 training frame counts at `E\v2\training_spec.json:254-318`, and records the tail contract with sequence, 1,367 image frames, 1,406 GT rows, exact GT SHA, and 39 ignored annotation-only tail rows at `E\v2\training_spec.json:319-326`.

The v2 trainer implements the exception in memory only for that sequence. It reads `count = spec['training_frame_counts'][name]`, requires `count==1367`, `len(info['bbox'])==1406`, and the exact GT SHA, then slices `info` to `[:count]`: `E\v2\train.py:44-51`. All sequences then must satisfy `len(info['bbox']) == count`, finite valid boxes, and enough visible frames: `E\v2\train.py:52-54`. The trainer also asserts every expected color/depth filename exists exactly from `00000001` through the sequence count before training starts: `E\v2\train.py:139-142`.

**4. GT provenance is real dataset GT, hash-bound; I found no invented-label path in the reviewed training entry.** The v2 spec points to the dataset root and the fixed fitting manifest at `E\v2\training_spec.json:6-10`, and contains 63 fit GT hashes starting at `E\v2\training_spec.json:100-164`. The trainer verifies every bound `groundtruth.txt` against those hashes before execution: `E\v2\train.py:137-138`, and repeats that GT hash verification after the loop: `E\v2\train.py:271-275`. Training samples come from `DepthTrack(root=spec['dataset_root'])` restricted to `spec['fit_sequences']`: `E\v2\train.py:41-43`. The actual search annotations used for loss are returned by `dataset.get_frames(...)` and passed as `search_anno`: `E\v2\train.py:101-112`; the loss then uses `data['search_anno']` as the target boxes at `E\v2\train.py:219-224`.

**5. The v1/v2 trainer change is minimal and targeted.** A local text diff between `E\train.py` and `E\v2\train.py` showed only two hunks: the `toy07` annotation-tail handling in `PairedClips.__init__`, and the preflight color/depth filename assertions before importing/building the model. The corresponding source evidence is v1’s original annotation checks at `E\train.py:41-49` and GT-hash checks ending at `E\train.py:127-131`, versus v2’s tail handling at `E\v2\train.py:44-54` and added frame-file checks at `E\v2\train.py:139-142`.

The paired split/seeds/budget/gates remain fixed by v2 spec: fit split `E\v2\training_spec.json:11-75`, development split `E\v2\training_spec.json:76-99`, seeds and training budget `E\v2\training_spec.json:165-174`, keep-rate/loss/sampler/initialization/arm-difference statements `E\v2\training_spec.json:199-217`, contract settings `E\v2\training_spec.json:220-228`, recursive gates `E\v2\training_spec.json:229-243`, and public progression only after the paired recursive gate at `E\v2\training_spec.json:244`.

**6. The v2 contracts are completed, but they are not completed training.** The dataset contract is explicit: `status=complete_dataset_contract`, bound trainer/spec hashes, and no network forwards or optimizer steps: `E\v2\dataset_contract.json:2-5` and `E\v2\dataset_contract.json:144-147`. Its examples include the old failure-adjacent range and the repaired `toy07` sample at index 164 with search IDs `[1318, 1319, 1366, 1338]`: `E\v2\dataset_contract.json:75-90`.

Both sampler contracts are complete two-step checks, with 8 microbatches, 16 clips, 64 search frames, no retained weight, no development evaluation, no public evaluation, and no adopted weight. Control evidence is `E\v2\sampler_contract\control\result.json:2-5`, `E\v2\sampler_contract\control\result.json:25-41`; clone evidence is `E\v2\sampler_contract\clone\result.json:2-5`, `E\v2\sampler_contract\clone\result.json:25-41`. Their data stream SHA is identical in both arms: `0616c78386ad3df13d7591a13417d28913d30ace77e2151d677fe3c7072d41c7`, at `E\v2\sampler_contract\control\result.json:29` and `E\v2\sampler_contract\clone\result.json:29`.

**7. Local byte bindings check out.** `E\revision_download_binding.json:58-80` binds the v2 preparation, spec, trainer, dataset contract, and launch preflight. `E\revision_download_binding.json:82-112` binds the v2 runner/queue/sampler-contract logs and exit files, and `E\revision_download_binding.json:114-160` binds both sampler-contract result trees. I independently recomputed byte counts and SHA-256 locally for all 40 entries in `revision_download_binding.json`; result: **40 matched, 0 missing, 0 mismatched**.

**8. Launch evidence is preflight/queue evidence, not completion evidence.** `E\v2\training_launch_preflight.json:2-13` binds the v2 spec, trainer, runner, queue script, dataset contract, contract result hashes, paired input equality, and source-hash checks. It records `control_launch_gpu=0`, clone status as queued after native full127 success and GPU1 release, and starting from the original base at `E\v2\training_launch_preflight.json:17-19`. The experiment plan makes the same distinction: two checks are separate from retained training at `P\diagnostics\m55\EXPERIMENT_PLAN.md:44-51`, full training and recursive validation are future sequential stages at `P\diagnostics\m55\EXPERIMENT_PLAN.md:53-66`, and the v2 section says control launched, clone queued, and no M55 accuracy gain is published before training and recursive validation complete at `P\diagnostics\m55\EXPERIMENT_PLAN.md:70-82`.

Action items:

1. Keep the launch/report wording strict: v1 produced **20 optimizer steps / 80 optimized microbatches**, not a completed training result. If mentioning `82`, call it 82 logged old batch rows, not 82 optimizer-completed batches.
2. For v2, claim only: dataset contract complete, sampler contracts complete, hashes locally bound, control launch preflight recorded, clone queued in the reviewed artifact. Do not claim clone training started, retained training completed, final weights exist, recursive metrics exist, or public benchmark gains.
3. After retained training completes, the next integrity checkpoint should verify final weight hashes, both arms’ full input-stream equality, step/sample counts, final exits, and then independently recompute the 22-sequence recursive metrics before any performance claim.

Supported claims:

- v1 control failed on a real unreadable `toy07_indoor_320/color/00001406.jpg` path after partial progress only.
- Clone v1 did not start before cancellation.
- The `toy07` exception is a prior, hash-bound 1,367-image / 1,406-GT annotation-tail contract.
- V2 adds a targeted in-memory tail handling and explicit frame-file checks.
- V2 dataset and sampler contracts completed, with no retained weights and no evaluation metrics.
- Local downloaded artifact hashes match the recorded binding.

Qualified claims:

- “V2 is ready to launch retained training” is supported only as a local contract/preflight conclusion, not as evidence of actual remote completion.
- “Same data stream in sampler contract” is supported for the 16-clip sampler contract, not for the full retained training run.

Unsupported in the reviewed artifacts:

- Any M55 tracking improvement.
- Any completed retained training result or final model weight.
- Any recursive low22 result.
- Any DepthTrack Test, CDTB, or VOT public benchmark result.
- Any independent live-server status.