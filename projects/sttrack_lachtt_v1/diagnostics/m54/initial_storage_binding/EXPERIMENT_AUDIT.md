Overall verdict: **WARN, pre-execution only**. I found **no concrete implementation defect** in the prepared M54 two-template reader, collection, training, recursive runner, analyzer wiring, or completed interface/head-wiring contracts. The warning is because the available artifacts still stop at **contract/runtime-contract evidence**: there is no completed long collection, trained reader checkpoint/result, recursive development result, low22 result, or public benchmark result available to verify.

This is a same-family GPT Type-A advisory review only, not a cross-family acquittal. I did not edit files, contact the server/network, use GPU, or launch experiments. I read the primary files directly and recomputed local SHA-256 bindings for the M54 plan/source/run files plus the provided native source snapshot.

Root audited: `C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1`

### A. Ground-truth provenance and separation: PASS

The collection path does not use dataset GT. It reads RGB-D frames from `dataset_root` and checks native predictions against sealed expected rows, which are prediction-derived consistency references rather than labels: `overlay/tools/collect_sttrack_m54.py:46-50` loads image/depth frames, and `overlay/tools/collect_sttrack_m54.py:65-68` compares `result['target_bbox']` and `result['best_score']` to `case['expected_rows'][frame]`. The collection receipt code records `labels_opened=False` and `optimizer_steps=0` at `overlay/tools/collect_sttrack_m54.py:95-99`; the completed contract artifact reports the same at `diagnostics/m54/contract.json:33-38`.

GT is opened only after sealed feature packets and identities are verified. Training checks the collection exit/receipt and all feature hashes before GT at `overlay/tools/train_sttrack_m54.py:23-40`, then explicitly opens `groundtruth.txt` at `overlay/tools/train_sttrack_m54.py:43-50`. Per-window GT labels are indexed by the recorded physical frame at `overlay/tools/train_sttrack_m54.py:53-58`. Invalid GT is excluded from supervised fitting through `valid` and `fit` at `overlay/tools/train_sttrack_m54.py:54-63`, and only valid indices are shuffled/trained at `overlay/tools/train_sttrack_m54.py:81-88`.

Recursive analysis likewise verifies the sealed recursive receipt and trajectory hashes before opening development GT: `overlay/tools/run_sttrack_m54.py:30-47` validates the recursive receipt, per-sequence trajectory hashes, row counts, initialization row, finite boxes, and reader choices; the comment at `overlay/tools/run_sttrack_m54.py:54` marks that both result families are verified before development GT; GT is then opened at `overlay/tools/run_sttrack_m54.py:56-62`.

The official benchmark boundary is clear. The plan says recursive analysis uses the existing continuous IoU/H10 proxy, not public DepthTrack/CDTB/VOT metrics: `diagnostics/m54/EXPERIMENT_PLAN.md:29-31`. The public launch is disabled in spec at `diagnostics/m54/spec.json:33`, and recursive result writing, if reached, marks `public_evaluation=False` at `overlay/tools/run_sttrack_m54.py:77-82`.

### B. Score normalization and denominators: PASS

I found no metric normalized by the model’s own maximum/minimum/mean output. The training overlap helper computes ordinary xywh IoU using intersection over union at `overlay/tools/train_sttrack_m42.py:19-23`. The recursive evaluator excludes initialization and invalid GT, then reports valid-frame IoU sum, mean IoU, low-IoU frames, H10 episodes, and invalid GT counts at `overlay/tools/analyze_sttrack_m42_recursive.py:20-30`.

M54 recursive aggregation uses valid-frame denominators and sequence means: `overlay/tools/run_sttrack_m54.py:63-68`. Gate calculations compare reader metrics to the default baseline rather than self-normalizing by reader statistics: `overlay/tools/run_sttrack_m54.py:69-76`.

The reader does use normalized geometry and log native score as model inputs, but these are not reported metrics. Geometry is normalized by image size at `overlay/lib/test/tracker/sttrack_template_reader.py:59-65`, and the model adds log native scores to logits at `overlay/lib/models/sttrack/lachtt_template_reader.py:33-36`.

### C. Result existence and execution status: WARN

Completed interface artifacts exist and are internally consistent, but no trained/performance result exists yet.

Available completed contract evidence:

- `diagnostics/m54/contract.exit:1` is `0`.
- `diagnostics/m54/contract.json:2-3` reports `status: PASS`.
- `diagnostics/m54/contract.json:29-38` reports 240 events, 240 frames, 3 native updates, 90 different read boxes, spec hash `2ec9bd39...`, `labels_opened=false`, `optimizer_steps=0`, and `plain_native_contract_exact=true`.
- `diagnostics/m54/runtime_contract.exit:1` is `0`.
- `diagnostics/m54/runtime_contract.json:2-8` reports `status: PASS`, spec hash `2ec9bd39...`, 58,923 parameters, finite/active gradient blocks, view-exchange check, initial logits equal native log scores, and exact actual `choose_view` input path.
- `diagnostics/m54/runtime_contract.json:9-31` reports two 120-frame contract sequences, 3 total native updates, `gt_opened=false`, `optimizer_steps=0`, and scope limited to “Synthetic head wiring and causal native-substitution interface contract; no performance result.”
- `diagnostics/m54/contract_controller.exit:1` is `0`.

No completed long-run evidence was available at audit time. The local diagnostics directory did not contain `collection_receipt.json`, `collection.exit`, `training_result.json`, `training.exit`, `reader_final.pth`, `recursive_receipt.json`, `recursive.exit`, `recursive_result.json`, `analysis.exit`, or `controller.exit`. Absence has no file:line location, but the prepared metadata also states that long collection and training had not been launched: `diagnostics/m54/preparation.json:7-9`.

The execution script is correctly ordered for the post-contract pipeline: collection, training, recursive run, and analysis are invoked in sequence at `diagnostics/m54/run.sh:8-24`, and the final controller exit is written at `diagnostics/m54/run.sh:26-27`. Normal collection requires both completed contracts before proceeding: `overlay/tools/collect_sttrack_m54.py:34-39`. The separate contract launcher runs collection-contract first and runtime-contract second at `diagnostics/m54/run_contract.sh:7-14`.

### D. Dead code, execution wiring, runtime defects: PASS

I found no dead metric or reader path in the prepared M54 pipeline.

The reader is wired into runtime selection. `STTrackTemplateReader` defines the exact reader fields at `overlay/lib/test/tracker/sttrack_template_reader.py:10`, moves those fields into the reader at `overlay/lib/test/tracker/sttrack_template_reader.py:29-31`, computes current and alternate native forwards at `overlay/lib/test/tracker/sttrack_template_reader.py:39-45`, constructs boxes/scores/maps/RoIs at `overlay/lib/test/tracker/sttrack_template_reader.py:46-65`, and commits the chosen state/query/previous-RoI at `overlay/lib/test/tracker/sttrack_template_reader.py:66-70`.

The two-view branch handles query mutation correctly. The native model mutates `track_query_before` inside `overlay/lib/models/sttrack/sttrack.py:111-132` and returns it at `overlay/lib/models/sttrack/sttrack.py:145-160`. M54 passes a cloned query state into each branch at `overlay/lib/test/tracker/sttrack_template_reader.py:39-42`, computes current and `[initial, initial]` alternate separately at `overlay/lib/test/tracker/sttrack_template_reader.py:44-45`, and commits only the chosen branch query at `overlay/lib/test/tracker/sttrack_template_reader.py:68-70`. The clone helper detaches and clones both RGB/depth query tensors at `overlay/lib/test/tracker/sttrack_lachtt_observation.py:67-72`.

Read/write separation is implemented. The alternate read uses `[self.z_dict[0], self.z_dict[0]]`, while the dynamic template is only written under the native update condition after the chosen branch is committed: `overlay/lib/test/tracker/sttrack_template_reader.py:72-76`. The plan explicitly states that `[initial, initial]` is an initial-view duplicate, not a new view, at `diagnostics/m54/EXPERIMENT_PLAN.md:3`, and says initial reads do not permanently overwrite the dynamic template at `diagnostics/m54/EXPERIMENT_PLAN.md:15`.

The native-equivalence contracts cover both current-only and forced initial-read paths. Current-only collection overrides selection to current at `overlay/tools/collect_sttrack_m54.py:26-28` and, in contract mode, compares exact bbox, score, templates, query, and template image against an independent native `STTrack` instance at `overlay/tools/collect_sttrack_m54.py:70-76`. Runtime contract forces initial reads at fixed frames at `overlay/tools/check_sttrack_m54.py:54-64`, simulates native slot substitution and default write behavior at `overlay/tools/check_sttrack_m54.py:65-68`, and checks exact bbox, score, templates, query, and template image at `overlay/tools/check_sttrack_m54.py:69-74`. It also exercises the actual reader input path independently of forced choices at `overlay/tools/check_sttrack_m54.py:77-82`.

The tensor shapes and loss are consistent on source inspection. The reader expects RoIs `B,2,2,16,768`, references with the same local-token layout, maps `B,2,2,16,16`, geometry `B,2,4`, and scores `B,2`; it projects RoIs/references to 32D, runs attention, builds 165D evidence, and returns two logits at `overlay/lib/models/sttrack/lachtt_template_reader.py:22-36`. The training script passes exactly those fields at `overlay/tools/train_sttrack_m54.py:78-79`, applies cross-entropy over two logits at `overlay/tools/train_sttrack_m54.py:86-88`, clips gradients at `overlay/tools/train_sttrack_m54.py:90-93`, saves one final checkpoint with spec/collection/base-checkpoint binding at `overlay/tools/train_sttrack_m54.py:100-103`, and verifies strict reload equality at `overlay/tools/train_sttrack_m54.py:105-113`.

The completed runtime contract reports the same parameter count as the source architecture: `diagnostics/m54/runtime_contract.json:4` reports 58,923 parameters; the parameterized blocks are defined at `overlay/lib/models/sttrack/lachtt_template_reader.py:6-20`.

### E. Scope, frame coverage, separation, seed count, claim strength: WARN

The intended scope is bounded and mostly well labeled. The spec declares 63 fitting sequences, 22 development sequences, 10,615 fit event windows, and 93,362 fit tracking steps at `diagnostics/m54/spec.json:11-15`. The plan says these are M44 DepthTrack Train fit/development splits with no sequence overlap, and that the development set has been used many times and cannot be described as fresh unseen testing: `diagnostics/m54/EXPERIMENT_PLAN.md:17-23`.

Collection uses only fit cases: `overlay/tools/collect_sttrack_m54.py:29-30`; training checks 63 fit cases and unique sequence coverage at `overlay/tools/train_sttrack_m54.py:27-40`; recursive evaluation uses only development cases and asserts 22 unique sequences at `overlay/tools/run_sttrack_m54.py:25-27`.

The training protocol is single-seed and final-epoch only. The plan fixes seed 2026, AdamW, learning rate, weight decay, batch size, gradient clipping, 20 epochs, no class weights, no threshold scan, and no dev-best epoch selection at `diagnostics/m54/EXPERIMENT_PLAN.md:23`. The code uses the same seed and deterministic sample-order generator at `overlay/tools/train_sttrack_m54.py:66-84`, asserts the expected optimizer-step formula at `overlay/tools/train_sttrack_m54.py:99`, and records sample-order and reload evidence at `overlay/tools/train_sttrack_m54.py:119-129`.

The recursive gate matches the stated development gate. The spec requires mean IoU gain at least 0.01, at least 3 positive sequences, fewer low frames, no episode increase, and successful-sequence protection at `diagnostics/m54/spec.json:26-32`. The analysis implements those exact checks at `overlay/tools/run_sttrack_m54.py:69-80`.

The warning is claim-scope, not implementation failure: there is no completed training or recursive result yet, and even a future pass would first support only a **single-seed, repeatedly used 22-sequence DepthTrack Train development proxy**. It would not by itself support public benchmark improvement, stability, or a mechanism-isolated local-token/rotation claim. The plan itself says the local matching and response encoder are not separated and that any success can only be attributed to the whole two-combination reader at `diagnostics/m54/EXPERIMENT_PLAN.md:33`. It also says low22/public evaluation only follows after the recursive gate at `diagnostics/m54/EXPERIMENT_PLAN.md:31` and `diagnostics/m54/EXPERIMENT_PLAN.md:37`.

### F. Evaluation type classification: PASS with limits

The completed M54 contract artifacts are **self_supervised_proxy / interface-contract** evidence. They use real RGB-D sequences and permitted initialization boxes, but no GT labels and no optimizer steps. Evidence: `diagnostics/m54/contract.json:29-38`, `diagnostics/m54/runtime_contract.json:27-31`, `overlay/tools/collect_sttrack_m54.py:70-76`, and `overlay/tools/check_sttrack_m54.py:69-82`. These contracts support native-equivalence and reader wiring only, not performance.

The planned M54 training is **real_gt fitting supervision** on DepthTrack Train fit GT, after sealed feature verification. Evidence: `overlay/tools/train_sttrack_m54.py:23-50` and target rule at `overlay/tools/train_sttrack_m54.py:61-65`. The static fitting metrics written by training, if generated, would remain a fitting diagnostic, as the result scope is explicitly “Fitting-only training; recursive development evaluation still required” at `overlay/tools/train_sttrack_m54.py:119-129`.

The planned M54 recursive development analysis is **real_gt proxy evaluation**: it uses dataset GT on the 22-sequence Train development split and the M42 continuous IoU/H10 proxy, not official public benchmark evaluation. Evidence: `overlay/tools/run_sttrack_m54.py:54-68` and `overlay/tools/analyze_sttrack_m42_recursive.py:20-30`.

The prior M53 capacity basis is a **privileged offline real-GT capacity diagnostic**, not M54 causal/runtime performance. M54 binds its design to M53 capacity through `diagnostics/m54/spec.json:5-6` and `overlay/tools/sttrack_m54_common.py:28`, while the M54 plan limits this to design motivation at `diagnostics/m54/EXPERIMENT_PLAN.md:3` and `diagnostics/m54/EXPERIMENT_PLAN.md:33`.

No M54 evidence here is `synthetic_proxy`, `simulation_only`, or `human_eval`.

### Source and hash binding

All M54 declared source/run hashes matched my local recomputation:

- The spec binds the experiment plan and new M54 source files at `diagnostics/m54/spec.json:34-45`.
- I recomputed matching SHA-256 values for `EXPERIMENT_PLAN.md`, `lachtt_template_reader.py`, `sttrack_template_reader.py`, `sttrack_m54_common.py`, `collect_sttrack_m54.py`, `train_sttrack_m54.py`, `run_sttrack_m54.py`, `check_sttrack_m54.py`, and `run.sh`.
- The provided native source snapshot `C:/Users/gb/.codex_remote_staging/m54_native_source.py` recomputed to `d67d551a612b80cee5b19a00f6fecd5d0f7ed0c907e800f452873afd684cc58f`, matching the parent M44 native tracker binding at `diagnostics/m44/spec.json:7-11`.

M54 source binding is anchored through `check_sources`: it reads the M54 spec, parent M44 spec, inserts the parent repository path, calls M44 `check_binding`, verifies the parent spec, parent inference inputs, M54 plan/source/run files, and M53 capacity hash at `overlay/tools/sttrack_m54_common.py:13-29`. M44 `check_binding` verifies the parent training binding, parent source hashes, and native checkpoint hash at `overlay/tools/train_sttrack_m44.py:21-26`.

Source-completeness limit: the local published overlay directory does not contain `overlay/lib/test/tracker/sttrack.py`, while `overlay/lib/test/tracker/sttrack_template_reader.py:3` imports `lib.test.tracker.sttrack`. The provided staging snapshot fills that native-source audit gap and matches the M44 hash, but I did not contact the remote repository, so I cannot independently verify the current remote path state beyond the copied artifacts and contract outputs.

### Concrete defects found

None in the prepared M54 implementation from this read-only audit.

The only material limits are evidence limits:

- No completed long collection, training, recursive development result, low22 result, or public result was available.
- No independent recomputation of M54 training/recursive metrics is possible until raw feature packets, checkpoint/result JSON, recursive trajectories, and relevant GT copies exist.
- The completed contracts support interface and head wiring only; they do not support performance.
- The development split is reused, single-seed, and proxy-metric scoped.

### Prioritized action items

1. Do not claim M54 performance, recursive improvement, public benchmark improvement, or stability until the missing long-run artifacts exist: `collection_receipt.json`, `training_result.json`, `reader_final.pth`, `recursive_receipt.json`, `recursive_result.json`, and their exit files.

2. After collection completes, independently verify 63 feature files against `collection_receipt.json`, confirm 10,615 physical windows, 93,362 tracking steps, unique `sequence@frame` keys, no GT opened during collection, and exact native prediction consistency.

3. After training completes, recompute from sealed feature packets and GT: valid/invalid window counts, alternate target count, optimizer steps `20 * ceil(valid_windows / 32)`, sample-order hash, changed tensors, strict reload equality, selected/current static diagnostic metrics, and GT hashes.

4. After recursive development completes, recompute per-sequence and aggregate metrics from raw trajectories and GT using the M42 `statistics` definition at `overlay/tools/analyze_sttrack_m42_recursive.py:20-30`; verify all five frozen gates at `overlay/tools/run_sttrack_m54.py:69-80`.

5. If the artifacts are meant to be locally reproducible without the remote source tree, include the native `lib/test/tracker/sttrack.py` file in the published local source bundle or keep the provided staged snapshot explicitly referenced, because the local overlay copy is absent while the wrapper imports it.

### Claim status

Supported now:

- “M54 is a prepared frozen-native STTrack plus two-view visual reader experiment with hash-bound source and plan.” Supported by `diagnostics/m54/spec.json:16-24`, `diagnostics/m54/spec.json:34-45`, and local hash recomputation.
- “The interface contracts completed for two fitting sequences / 240 frames and used no GT or optimizer steps.” Supported by `diagnostics/m54/contract.json:29-38`, `diagnostics/m54/runtime_contract.json:27-31`, and exits at `diagnostics/m54/contract.exit:1`, `diagnostics/m54/runtime_contract.exit:1`, `diagnostics/m54/contract_controller.exit:1`.
- “Current-only M54 collection can preserve native STTrack behavior under the contract path.” Supported by source checks at `overlay/tools/collect_sttrack_m54.py:70-76` and contract result at `diagnostics/m54/contract.json:29-38`.
- “Forced initial-slot reading commits bbox, score, query, template state, and template image consistently with a native slot-substitution reference under the runtime contract.” Supported by `overlay/tools/check_sttrack_m54.py:60-82` and `diagnostics/m54/runtime_contract.json:9-31`.

Qualified:

- “M54 trains a reader using real GT.” Qualify as: real DepthTrack Train fit GT is opened only after sealed feature verification, invalid windows are excluded, and this is fitting supervision, not public evaluation. Evidence: `overlay/tools/train_sttrack_m54.py:23-65`.
- “M54 evaluates recursively.” Qualify as: prepared to evaluate all 22 repeated Train development sequences with continuous IoU/H10 proxy after trajectory sealing; no recursive result exists yet. Evidence: `overlay/tools/run_sttrack_m54.py:25-82` plus absence of result artifacts.
- “M54 uses an initial-template alternative.” Qualify as: it substitutes the dynamic slot with the initialization tensor, so `[initial, initial]` is not a new physical view or independent memory bank. Evidence: `diagnostics/m54/EXPERIMENT_PLAN.md:3`, `overlay/lib/test/tracker/sttrack_template_reader.py:25-27`, and `overlay/lib/test/tracker/sttrack_template_reader.py:44-45`.

Unsupported now:

- Any M54 trained-model performance number.
- Any M54 recursive development improvement.
- Any low22, DepthTrack Test, CDTB, VOT/full127, or official benchmark claim.
- Any claim that the reader is stable across seeds.
- Any claim that a local-token module, rotation behavior, or visual matching alone caused an improvement.
- Any claim that M54 fixes the M52 data-effect/recursive-state issue before the M54 recursive gate actually passes.