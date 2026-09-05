# M52 native source follow-up audit

GPT-5.5 xhigh, independent-context GPT-family Type-A advisory review. Reviewer response preserved verbatim; no cross-family acquittal.

Follow-up verdict: **source-completeness warning is largely resolved for the base tracker source**, but overall M52 evidence remains **WARN** because completed collection/training/recursive result artifacts are still not present in the local M52 bundle. This is an independent-context GPT-family Type-A advisory review, not a cross-family acquittal.

**Source Snapshot Integrity: PASS**

- Computed bytes for `diagnostics/m52/source_snapshots/sttrack.py`: **10,216 bytes**.
- Computed SHA-256: `d67d551a612b80cee5b19a00f6fecd5d0f7ed0c907e800f452873afd684cc58f`.
- `diagnostics/m52/source_snapshots/README.md:3-8` states this is a byte-for-byte snapshot from `/root/autodl-tmp/rgbd_baselines/STTrack_lachtt_v1/lib/test/tracker/sttrack.py`, with the same SHA and size, matching the M44 binding.
- `diagnostics/m44/spec.json:7-8` binds `lib/test/tracker/sttrack.py` to the same SHA.
- `diagnostics/m52/source_snapshots/README.md:10-13` explicitly says the snapshot is source-review evidence, not a new algorithm/change, and that `overlay/` is not a standalone upstream distribution.

Concrete conclusion: the previously missing base tracker source can now be audited from `source_snapshots/sttrack.py`, and its bytes match the M44 source binding. The local `overlay/lib/test/tracker/sttrack.py` import target remains absent from the overlay itself, but the README explicitly frames the overlay as non-standalone runtime overlay rather than a complete source tree.

**Initialization And Reset: PASS**

- Base `STTrack.__init__` loads the checkpoint, moves the network to CUDA/eval mode, initializes `state=None`, sets `num_template`, feature size, output window, update interval/threshold, `debug`, `frame_id=0`, and `save_all_boxes`: `diagnostics/m52/source_snapshots/sttrack.py:20-62`.
- Base `initialize()` rebuilds the initial template from `init_bbox`, sets `self.z_dict = [template] * self.num_template`, clears `box_mask_z`, clears `track_query_before`, sets `state` to `init_bbox`, and resets `frame_id=0`: `diagnostics/m52/source_snapshots/sttrack.py:63-89`.
- M52 candidate-set subclass calls base `initialize()` and then resets its own association state: `reference_bank`, `previous_set=None`, and `previous_choice=0`: `overlay/lib/test/tracker/sttrack_candidate_set.py:19-22`.
- M52 collection calls `tracker.initialize(...)` at the start of every sequence, and also initializes the unhooked comparison tracker in contract mode: `overlay/tools/collect_sttrack_m52.py:83-86`.

Concrete conclusion: the source supports per-sequence reset of base query/template/state and candidate-set association state. I found no cross-sequence carryover defect in the reviewed initialization path.

**Native Template / Query / State Behavior: PASS**

- Base tracking samples the search crop from current `self.state`, processes it, conditionally passes `track_query_before` only after it exists, then updates `self.track_query_before` from the network output: `diagnostics/m52/source_snapshots/sttrack.py:95-115`.
- Base default box generation uses windowed score map, `box_head.cal_bbox`, mean predicted box, `map_box_back`, and `clip_box`: `diagnostics/m52/source_snapshots/sttrack.py:117-126`.
- Base dynamic template update occurs only when `num_template > 1`, update interval matches, and confidence exceeds threshold; it appends a new template and pops index 1 when over capacity: `diagnostics/m52/source_snapshots/sttrack.py:128-138`.
- Candidate-set tracker follows the same recursive state structure, but requests candidate features, observes candidates before association, uses default candidate 0 when there is no previous set, otherwise calls the association head with current/previous RoIs, references, geometry, scores, and `previous_choice`: `overlay/lib/test/tracker/sttrack_candidate_set.py:24-41`.
- Candidate-zero path uses native `box_head.cal_bbox` and max response confidence: `overlay/lib/test/tracker/sttrack_candidate_set.py:42-44`.
- Nonzero selected candidates use the selected candidate grid cell’s offset/size and confidence: `overlay/lib/test/tracker/sttrack_candidate_set.py:45-49`.
- State update, template update, reference-bank update, and previous-set/previous-choice update happen in that order after the decision: `overlay/lib/test/tracker/sttrack_candidate_set.py:50-57`.

Concrete conclusion: M52’s tracker source behaves as a causal recursive policy: candidate selection changes the next state, template update uses the post-decision state, and `previous_choice` is updated after each frame. I found no concrete reset/template/query/state logic defect in the reviewed files.

Qualification: the M52 contract compares observed-vs-unobserved `STTrackCandidateSet`, not `STTrack` base output. That is appropriate for M52 because the frozen collection policy is explicitly `Frozen M45 STTrackCandidateSet` in `diagnostics/m52/spec.json:4`, but it should not be cited as proof that `STTrackCandidateSet` is identical to native `STTrack` under every condition.

**State Capture Consistency: PASS, with small audit-scope qualification**

- M52 binding checks parent spec, inference input SHA, collector source SHA, policy checkpoint SHA, M45 training result SHA, and M51 recursive result SHA before running: `overlay/tools/collect_sttrack_m52.py:30-38`.
- Collection restricts cases to fit split and asserts 63 sequences / 1,511 events: `overlay/tools/collect_sttrack_m52.py:41-43`.
- Non-contract collection requires prior contract PASS before full capture: `overlay/tools/collect_sttrack_m52.py:44-48`.
- The forward wrapper captures the actual network output when the current frame is needed and returns the original output unchanged: `overlay/tools/collect_sttrack_m52.py:58-64`.
- The association pre-hook captures the actual association inputs on event frames: `overlay/tools/collect_sttrack_m52.py:66-71`.
- Contract mode instantiates an unhooked `STTrackCandidateSet` and checks exact output equality, template equality, and query equality against the hooked/observed path: `overlay/tools/collect_sttrack_m52.py:53-54` and `103-107`.
- The collector records pre-frame `prior`, `previous_set`, `previous_choice`, and dynamic template pointer before calling `track()`: `overlay/tools/collect_sttrack_m52.py:96-102`.
- It reconstructs candidate-zero/default output from the captured head using the pre-frame state, and asserts equality when the association choice is 0: `overlay/tools/collect_sttrack_m52.py:110-124`.
- At event frames, it asserts the association input’s `previous_choice`, current RoIs, previous RoIs, and concatenated geometry match the current/previous sets being recorded: `overlay/tools/collect_sttrack_m52.py:126-133`.
- It stores the actual association-input tensors, current/previous candidate boxes, candidate-zero/default boxes, selected output boxes, previous selected boxes, and per-event choices: `overlay/tools/collect_sttrack_m52.py:133-143`.
- It asserts finite tensors and positive geometry sizes before saving: `overlay/tools/collect_sttrack_m52.py:147-150`.
- Full non-contract collection asserts 1,511 events, 93,362 frames, and 63 receipts before writing `collection_receipt.json`: `overlay/tools/collect_sttrack_m52.py:164-175`.

Concrete conclusion: the capture path records the actual tensors passed to the association head, not a later reconstruction, and it cross-checks the key state links: current/previous RoIs, geometry, candidate-zero default box, selected output box, and `previous_choice`.

Qualification: the code explicitly asserts RoIs, geometry, and `previous_choice` against reconstructed/current state. It does not independently recompute and assert `scores` or `references` against a second source in these lines. Since those fields are captured directly from the live association pre-hook, this is not a concrete integrity defect, but it is a narrower consistency check than “all input fields independently recomputed and matched.”

**Pipeline Implementation Record: PASS for documentation honesty**

- `diagnostics/m52/PIPELINE_IMPLEMENTATION.md:3-5` says the implementation was prepared before M52 fitting-label analysis or optimizer steps and does not claim completed training or performance.
- `diagnostics/m52/PIPELINE_IMPLEMENTATION.md:9-21` documents the paired setup, including same physical events, logical views not independent events, separate candidate-zero/selected-box semantics, actual `previous_choice`, and matched initialization/order digests.
- `diagnostics/m52/PIPELINE_IMPLEMENTATION.md:25-31` documents the intended pipeline and states tracking uses no development GT until sealed trajectories are analyzed; it also says this is repeated DepthTrack Train development, not official DepthTrack Test/CDTB/VOT scoring.
- `diagnostics/m52/PIPELINE_IMPLEMENTATION.md:39-41` says AST/import/help/bash checks do not establish completed audit, training, loader equality, or performance improvement.
- `diagnostics/m52/PIPELINE_IMPLEMENTATION.md:45-49` preserves the original advisory WARN context and says completion receipts will be added as actually produced.
- `diagnostics/m52/PIPELINE_IMPLEMENTATION.md:51-53` says the tracker snapshot was subsequently supplied as evidence without modifying the running tracker/collection.

Concrete conclusion: the implementation documentation is appropriately conservative and does not overclaim execution.

**Concrete Defects Found In This Follow-Up**

No new concrete implementation defect was found in the added source snapshot, initialization/reset logic, native template/query/state behavior, or M52 state capture consistency path.

The only concrete source-packaging issue remains qualified rather than fatal: `overlay/lib/test/tracker/sttrack_candidate_set.py:3` imports `lib.test.tracker.sttrack`, while the local overlay path previously lacked `overlay/lib/test/tracker/sttrack.py`. The new source snapshot resolves auditability of that file, and `diagnostics/m52/source_snapshots/README.md:12-13` clarifies that `overlay/` is not intended as a standalone upstream distribution. If someone tries to run from the local overlay alone, the import target is still not present at the overlay path; that is separate from the remote runtime binding.

**Remaining Evidence Limitations Preserved From Original Review**

- No local M52 completion artifacts are present for completed collection/training/recursive evaluation: `collection_receipt.json`, `data_audit.json`, `runtime_contract.json`, `control/training_result.json`, `mixed/training_result.json`, recursive shard receipts, `recursive_result.json`, and `per_sequence.csv` remain unavailable in the local M52 bundle I audited.
- `diagnostics/m52/PIPELINE_IMPLEMENTATION.md:39-41` explicitly says the existing preparation checks do not prove completed data audit, training, loader equality, or performance improvement.
- `diagnostics/m52/PIPELINE_IMPLEMENTATION.md:48-49` confirms only launch/contract evidence was available at review time and completion receipts are future/evidence-dependent.
- Official public benchmark evidence remains absent. The implementation doc says the recursive evaluation is not official DepthTrack Test, CDTB, or VOT scoring: `diagnostics/m52/PIPELINE_IMPLEMENTATION.md:28-31`; M44 public gate still requires later frozen low22 and three-dataset validation: `diagnostics/m44/spec.json:101`.

**Updated Claim Status**

Supported:
- The exact base `sttrack.py` source is now available for audit and byte-matches the M44 source binding.
- M52 per-sequence initialization/reset is correctly represented in the reviewed source.
- M52 capture records actual association-head inputs and checks key state consistency against the running tracker state.
- The 240-frame observer-vs-unobserved candidate-set contract is meaningful for confirming the capture hook does not perturb the frozen M45 candidate-set path.

Qualified:
- Source completeness is sufficient for reviewing `sttrack.py`, but the local overlay remains non-standalone by design.
- The M52 contract should be described as hooked-vs-unhooked `STTrackCandidateSet` equality, not as native `STTrack` equivalence.

Unsupported until result artifacts exist:
- Completed M52 collection.
- Any M52 paired training result.
- Any M52 recursive development gain.
- Any `primary_pass`, `advancing_arm`, or state-data benefit over paired control.
- Any public benchmark improvement claim.