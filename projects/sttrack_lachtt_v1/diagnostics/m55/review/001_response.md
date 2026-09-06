M55 preflight review complete. This is a read-only same-GPT-family Type-A advisory review. I did not edit files, use SSH/GPU, launch jobs, or infer remote completion from local log absence.

Path aliases used below:

- `S = C:\Users\gb\.codex_remote_staging`
- `E = C:\Users\gb\.codex_remote_staging\m55_training_evidence_20260906`

Overall verdict: **WARN**, not FAIL. I found no hard implementation blocker in the listed retained-training path. The main pre-launch issue is a **contract-documentation mismatch**: `training_spec.json` still says its contract has 2 optimizer steps, while the completed `contract_v2` checker/results are explicitly 3 optimizer steps. The existing three-step contract is useful evidence, but it is not full retained training, not final recursive validation, and not a performance result.

| Check | Status | Evidence | Finding |
|---|---:|---|---|
| A. Dataset GT provenance and fit/development isolation | PASS | `S\prepare_sttrack_m55_training.py:18-21`, `S\prepare_sttrack_m55_training.py:30-31`, `E\training_spec.json:8-10`, `E\training_spec.json:11-76`, `E\training_spec.json:76-164`, `S\train_sttrack_m55.py:41-49`, `S\train_sttrack_m55.py:127-130`, `E\code\control\lib\train\dataset\depthtrack.py:86-104` | The fitting/development split is derived from the frozen fitting manifest; preparation asserts 63 fit and 22 development sequences with no overlap. Training uses only `spec['fit_sequences']`. GT is loaded from DepthTrack `groundtruth.txt`, with valid/visible defined from dataset boxes. |
| B. Same initialization, sample/order budget, intended arm difference | PASS | `S\train_sttrack_m55.py:142-150`, `S\train_sttrack_m55.py:151-155`, `S\train_sttrack_m55.py:183-187`, `S\train_sttrack_m55.py:257-258`, `E\training_spec.json:165-174`, `E\training_spec.json:212-214`, `E\preparation.json:325-326`, `E\code\control\lib\models\sttrack\sttrack.py:94-98`, `E\code\clone\lib\models\sttrack\sttrack.py:94-98` | Both arms use the same model seed, strict full checkpoint load, same indexed per-sample seeds, same epochs/samples/accumulation budget, and the only declared source change is `temp_x_flip=temp_x.clone()` and `temp_r_flip=temp_r.clone()` in the clone arm. I independently diffed the two listed `sttrack.py` snapshots and found only those two assignment changes. |
| C. Original sampler/processing parity, temporal order, invalid data | PASS | `E\code\control\lib\train\data\sampler.py:61-93`, `E\code\control\lib\train\data\sampler.py:123-137`, `S\train_sttrack_m55.py:67-104`, `E\code\control\lib\train\data\processing.py:81-85`, `E\code\control\lib\train\data\processing.py:100-131`, `E\training_spec.json:209` | M55 reproduces the original causal-with-replacement sampling pattern: base template, one previous template, four later search frames, no sorting/deduplication. It retains the native processing path and invalid-crop resampling via `data['valid']`. It intentionally omits the original sampler’s broad exception swallowing; that matches the spec and is preferable for fail-fast training. |
| D. Accumulation, loss, keep-rate, scheduler, checkpoint/error semantics | PASS with WARN limits | `S\train_sttrack_m55.py:156-162`, `S\train_sttrack_m55.py:188-189`, `S\train_sttrack_m55.py:206-224`, `S\train_sttrack_m55.py:238-255`, `S\train_sttrack_m55.py:267-282`, `E\code\control\lib\utils\ce_utils.py:63-70`, `E\training_spec.json:192-208`, `E\training_spec.json:217-219` | The retained trainer sums four search-frame losses, divides by gradient accumulation before backward, clips at 0.1, steps every four microbatches, uses StepLR after each epoch, and writes a rolling optimizer checkpoint plus final model-only weights. Errors are not hidden by a broad training try/except; nonfinite loss/grad assertions should stop the run. Limit: the rolling checkpoint is written but the script has no automatic resume/load path, so it should not be described as resumable unless an external process handles that. |
| E. Three-step contract versus retained training/recursive metrics | WARN | `E\contract_v2\control_result.json:402-419`, `E\contract_v2\clone_result.json:402-419`, `E\contract_v2\control_result.json:452-483`, `E\contract_v2\clone_result.json:452-483`, `E\contract_v2\control_result.json:908-911`, `E\contract_v2\clone_result.json:908-911`, `E\training_spec.json:220-228`, `E\check_training_v2.py:100-125`, `E\check_training_v2.py:142-152` | The completed contract evidence is a three-step real-data full-network optimizer/memory check on two fitting clips, with no saved trained checkpoint, no public evaluation, and no development GT use. It is not retained training or recursive validation. Concrete mismatch: `training_spec.json` still records its `contract.optimizer_steps` as `2`, while `contract_v2` results and checker are `3` optimizer steps. |
| F. Claims that may/may not be made | WARN | `E\training_spec.json:215-244`, `E\contract_v2\control_result.json:908-911`, `E\contract_v2\clone_result.json:908-911`, `S\check_training_v2.py:1`, `S\check_training_v2.py:151-152` | Supported now: prepared paired training design, strict base initialization path, listed source hash checks, and a completed three-step real-data contract for both arms. Unsupported now: retained training completion, recursive improvement, public benchmark performance, or any current remote completion claim. |
| G. Minimal actionable issues before retained launch | WARN | See issues below | No hard blocker found. Two small issues should be handled by labeling/documentation before launch. |

Hash checks I performed:

- `E\training_spec.json`: SHA-256 `e3f62b46e578e896be9dd7cd23d32d5fefa1f231d2f260dec7e241d4213489ce`.
- `S\train_sttrack_m55.py`: SHA-256 `29df742a151451057fc1b9a62cd75b93efa20de3d407720dc29fe6d188066b2c`, matching `E\training_spec.json:5`.
- `E\preparation.json`: SHA-256 `d830a16cb5654e2c85936cc930512086b901c7c8221e4a10a8ffbfdf175024ef`, matching `E\training_spec.json:4`.
- Listed source snapshot hashes matched `preparation.json`: control `sttrack.py`, clone `sttrack.py`, control sampler, processing, DepthTrack dataset, actor, `ce_utils.py`, and `deep_rgbd_256_lachtt_v1.yaml`. Relevant declared hashes include `E\preparation.json:7-24`, `E\preparation.json:48`, `E\preparation.json:165`, `E\preparation.json:207`, and the listed-file hashes also match the values printed from the local files.
- The full `preparation.json` declares 157 source entries per arm, but this local evidence bundle contains only the subset listed in the request. I verified the listed files, not the entire declared source tree. The retained trainer itself would assert all remote code hashes at runtime via `S\train_sttrack_m55.py:124-126`.

Contract parity findings:

- Both contract results are `complete_training_contract`: `E\contract_v2\control_result.json:2-8`, `E\contract_v2\clone_result.json:2-8`.
- Both use the same sequences and frames: `chair01_indoor`, `cube04_indoor`, frames `0..4` at `E\contract_v2\control_result.json:402-412` and `E\contract_v2\clone_result.json:402-412`.
- Both use batch size 2, four search frames, three optimizer steps, keep-rate `[1.0]`, and 133,065,093 trainable parameters at `E\contract_v2\control_result.json:413-419` and `E\contract_v2\clone_result.json:413-419`.
- Both have identical `input_tensor_sha256` and raw input hashes beginning at `E\contract_v2\control_result.json:420-430` and `E\contract_v2\clone_result.json:420-430`; I compared the full objects and they match.
- Both have identical initial-state hash objects, gradient-group counts, missing-gradient lists, and changed-tensor sets in my parsed comparison. The visible gradient-group sections match at `E\contract_v2\control_result.json:486-512` and `E\contract_v2\clone_result.json:486-512`.
- The losses differ between arms, which is expected after the intended source change: control records are at `E\contract_v2\control_result.json:452-483`, clone records at `E\contract_v2\clone_result.json:452-483`.

Important implementation detail: the original actor source still contains a broad `try/except` around GIoU at `E\code\control\lib\train\actors\sttrack.py:109-113`, but the retained M55 trainer does not use that actor path for loss. It imports `giou_loss`, `FocalLoss`, and computes/validates loss directly at `S\train_sttrack_m55.py:134-137` and `S\train_sttrack_m55.py:206-224`. That avoids hiding training loss errors in the retained path.

Minimal actionable issues before launch:

1. **Label or update the contract step count.** `E\training_spec.json:220-228` says the spec contract has `optimizer_steps: 2`, but the completed `contract_v2` artifacts and checker are three-step: `E\check_training_v2.py:100-125`, `E\check_training_v2.py:147`, `E\contract_v2\control_result.json:415`, and `E\contract_v2\clone_result.json:415`. This is not a training-code blocker, but it is a concrete documentation/binding mismatch.

2. **Do not describe the rolling checkpoint as automatic resume.** The retained trainer saves optimizer/model/RNG/scheduler state to `latest.pth` at `S\train_sttrack_m55.py:247-255`, but no resume/load path is present in the listed trainer. The spec already says failures should end with nonzero exit and no automatic relaunch at `E\training_spec.json:217`; keep that wording.

3. **Keep the contract scope narrow.** The existing contract saved no trained checkpoint and used no development GT: `E\contract_v2\control_result.json:908-911` and `E\contract_v2\clone_result.json:908-911`. It supports “optimizer/memory contract passed on two fitting clips,” not retained training completion or recursive gains.

Supported pre-launch claims:

- M55 has a frozen paired training spec with 63 fit and 22 development sequences, no fit/dev overlap at preparation time, and a fixed 15-epoch, 2,048-sample-per-epoch, 3,840-step-per-arm retained budget.
- The retained trainer strictly loads the base checkpoint and verifies all loaded tensors before training.
- The intended arm difference is limited to the two clone assignments in `lib/models/sttrack/sttrack.py`.
- The contract_v2 evidence shows both arms can run three real-data optimizer steps with identical initial state and identical input tensors, and no retained contract weights were adopted.

Unsupported at this point:

- Any claim that retained training has completed.
- Any claim that M55 improves recursive tracking.
- Any public DepthTrack/CDTB/VOT claim.
- Any inference about current remote status from local logs being absent or present outside the listed artifacts.