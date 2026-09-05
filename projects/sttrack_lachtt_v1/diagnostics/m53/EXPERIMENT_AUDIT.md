# M53 contract and launch audit

Full independent-context reviewer response preserved verbatim. The review concerns the listed contract/launch artifacts, not a completed capacity result.

Read-only M53 audit completed. This is a GPT-family Type-A advisory review only, not a cross-family acquittal. I did not edit files or launch experiments.

Overall verdict: **WARN**. I found **no concrete defect** in the frozen historical-template capacity plan, collector logic, completed purity contract, or launch binding. The warning is because only the purity contract and launch observation are complete in the provided artifacts; the full collection receipt, GT-based capacity analysis, capacity arithmetic, and any downstream/public result remain unverified.

### A. Ground-truth provenance — PASS, with future-analysis limit

M53 is explicitly a GT-free collection followed by later offline capacity analysis. The spec declares `gt_read_during_collection: false` and `optimizer_steps: 0` at `diagnostics/m53/spec.json:81-82`. The plan says the collector reads images, initialization boxes, and sealed predictions, “never subsequent GT or labels” at `diagnostics/m53/EXPERIMENT_PLAN.md:22-27`, and opens GT only after sealing outputs at `diagnostics/m53/EXPERIMENT_PLAN.md:45-51`.

The collector code matches that: it loads fit cases from sealed M44 `inference_inputs.json` at `overlay/tools/collect_sttrack_m53.py:33-44`, reads RGB-D frames at `overlay/tools/collect_sttrack_m53.py:94-105`, and checks sealed native expected rows at `overlay/tools/collect_sttrack_m53.py:117-120`. I saw no GT read in the collector path. The completed contract records `labels_opened: false` at `diagnostics/m53/contract.json:36`.

Limit: the planned post-seal capacity analysis is not present in the provided files, so its actual GT handling is not yet verifiable.

### B. Score normalization — PASS for inspected artifacts

No self-referential normalization is present in the inspected contract/collector artifacts. The contract reports counts and raw tolerances: events, frames, past-template shadows, native-template updates, max bbox error, max score error, and exactness booleans at `diagnostics/m53/contract.json:29-37`. The planned capacity screen uses raw IoU thresholds, not prediction-derived denominators: severe default IoU `<= 0.1`, past top1 IoU `>= 0.5`, at least 10 events and 3 sequences at `diagnostics/m53/spec.json:88-94`.

Limit: no completed GT capacity result file was provided, so result-level denominator arithmetic cannot yet be audited.

### C. Result existence and execution status — WARN

The purity contract exists and completed: `contract.exit` is `0` at `diagnostics/m53/contract.exit:1`, and `contract.json` reports `status: PASS` at `diagnostics/m53/contract.json:1-2`. The two contract sequences ran 240 frames total, with 110 past-template shadows, 3 native template updates, exact current-template replay, unchanged public state, exact plain-native contract, no labels, and zero optimizer steps at `diagnostics/m53/contract.json:29-40`.

The launch binding exists but is not a completed collection result. It records expected full-run workload, not measured completion: 14,349 expected past-template shadows and 109,222 expected forward calls at `diagnostics/m53/execution_binding.json:8-9`, with `public_evaluation: false` and `automatic_training: false` at `diagnostics/m53/execution_binding.json:10-11`. The launch observation shows a running controller/child process, GPU memory use, `completed_sequence_files: 0`, and `terminal_files: []` at `diagnostics/m53/launch_observation.json:2-7`. `run.sh` would write `collection.exit` and `controller.exit` only after the collector exits at `diagnostics/m53/run.sh:5-8`.

No completed full collection receipt or capacity result was provided, so no M53 capacity outcome can be claimed yet.

### D. Dead code / execution wiring — PASS for contract and collector; WARN for absent capacity result

The relevant collector checks are wired. Source/spec bindings are checked at `overlay/tools/collect_sttrack_m53.py:33-40`; the two-sequence contract mode is selected at `overlay/tools/collect_sttrack_m53.py:45-50`; non-contract mode requires the passed contract first at `overlay/tools/collect_sttrack_m53.py:47-50`. The contract also compares against a plain native tracker, including output equality, templates, query state, and template patch array at `overlay/tools/collect_sttrack_m53.py:122-127`.

Native-path preservation is explicitly checked: the collector verifies sealed native bbox/score against expected predictions at `overlay/tools/collect_sttrack_m53.py:117-120`, decodes the baseline and asserts it equals the tracker result at `overlay/tools/collect_sttrack_m53.py:130-133`, hashes state before shadow forwards at `overlay/tools/collect_sttrack_m53.py:134`, and requires unchanged state after current-template replay and each historical shadow at `overlay/tools/collect_sttrack_m53.py:143-152`.

The full capacity analysis stage is not available, so its metric wiring remains unverified.

### E. Scope and claim strength — WARN

The completed contract is a **purity/interface contract**, not a performance test. The plan says this explicitly at `diagnostics/m53/EXPERIMENT_PLAN.md:38-43`. The completed contract covers only chair01 and cube04, 240 frames, 238 events, and 110 past-template shadows at `diagnostics/m53/contract.json:3-31`.

The intended main M53 scope is 63 fit sequences and 1,511 events, documented at `diagnostics/m53/spec.json:76-78` and `diagnostics/m53/EXPERIMENT_PLAN.md:14-20`, but the launch observation does not show completion. There is no automatic training or public evaluation: `diagnostics/m53/spec.json:95-96`, `diagnostics/m53/EXPERIMENT_PLAN.md:54-60`, and `diagnostics/m53/execution_binding.json:10-11`.

A valid current claim is limited to: “the collector purity contract passed for two fitting sequences, and the full collection was launched.” Any capacity, reader-design, tracker-advancement, training, or public benchmark claim is premature.

### F. Evaluation classification — WARN

- Purity contract: **self_supervised_proxy / interface contract**. It uses sealed native predictions and state equality, not GT performance. Evidence: `diagnostics/m53/EXPERIMENT_PLAN.md:38-43`, `diagnostics/m53/contract.json:29-37`.
- Planned historical-template capacity analysis: **real_gt offline capacity diagnostic**, with privileged components. The plan says GT is opened after sealing and that restricting historical writes by historical GT IoU and choosing the best view are privileged diagnostics, never deployment inputs at `diagnostics/m53/EXPERIMENT_PLAN.md:45-51`.
- Public evaluation: **unverified / unavailable**. Public automatic launch is false at `diagnostics/m53/spec.json:95`.

### Specific integrity checks

Strictly past template use is implemented. The archive starts with initialization at `overlay/tools/collect_sttrack_m53.py:105`; frame-t events assert the active archive frame is before the current frame and iterate only over `archive[:-1]` for alternatives at `overlay/tools/collect_sttrack_m53.py:147-151`; frame-t writes are appended only after all frame-t counterfactuals at `overlay/tools/collect_sttrack_m53.py:158-162`.

Query mutation is real and handled. The model mutates the supplied `track_query_before` object by slicing and concatenating into it at `overlay/lib/models/sttrack/sttrack.py:111-132`, then returns it at `overlay/lib/models/sttrack/sttrack.py:147`. The collector captures cloned query state at `overlay/tools/collect_sttrack_m53.py:61-65`, and each shadow receives its own cloned query state at `overlay/tools/collect_sttrack_m53.py:136-141`. The clone helper detaches and clones both query tensors at `overlay/lib/test/tracker/sttrack_lachtt_observation.py:67-72`.

Same-state current-template replay is checked. The collector replays `capture['templates'][1]` as the current-template control and requires identical score, size, and offset maps at `overlay/tools/collect_sttrack_m53.py:143-144`; the contract reports `current_template_replay_exact: true` at `diagnostics/m53/contract.json:33`.

Source binding is adequate for the inspected files. M53 binds collector and M41 helper hashes at `diagnostics/m53/spec.json:83-87`, and launch binds spec, contract, and controller hashes at `diagnostics/m53/execution_binding.json:3-5`. M44 binds the native tracker/model/observation sources at `diagnostics/m44/spec.json:7-10`; the provided native tracker snapshot corresponds to that bound native tracker source, but the live remote runtime file itself is not separately inspected here.

Concrete defects found: **none** in the inspected M53 plan, collector, completed purity contract, and launch binding.

Remaining unverified: completed 63-sequence collection receipt, sealed event files, GT-based capacity analysis, capacity threshold pass/fail arithmetic, any learned reader/training stage, and any public/official benchmark result.