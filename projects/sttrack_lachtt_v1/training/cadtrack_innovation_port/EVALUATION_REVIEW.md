Static review only, limited to the seven named files. No files changed; no tests, training, inference, evaluation, or network calls ran.

1. **P1: The frozen VOT execution-manifest hash is never enforced.**  
   [`prepare_evaluation.py:315`](C:/Users/gb/projects/Track_inspection_20261101/projects/sttrack_lachtt_v1/training/cadtrack_innovation_port/prepare_evaluation.py:315) records `execution_sha256` in the selection. However, `run_evaluation.py:191–200` verifies only the plan hashes, and [`run_evaluation.py:95`](C:/Users/gb/projects/Track_inspection_20261101/projects/sttrack_lachtt_v1/training/cadtrack_innovation_port/run_evaluation.py:95) trusts the loaded `execution.json` to supply its own source hashes. Neither path compares it with the selected execution digest.

   Consequently, replacing a VOT workspace together with its execution ledger can pass these checks while retaining the original selected final’s provenance. This matters because the wrapper, shard configuration, and manifest are bound through that ledger. Compare `execution.json` against the recorded selection digest before trusting its entries or launching shards.

2. **P2: An accepted VOT manifest can fail during merge, after all tracking has finished.**  
   [`run_evaluation.py:144`](C:/Users/gb/projects/Track_inspection_20261101/projects/sttrack_lachtt_v1/training/cadtrack_innovation_port/run_evaluation.py:144) unconditionally reads `manifest["source_sequences_root"]`. Preparation validates and uses the per-sequence `manifest["source"][name]["root"]` mapping, but never requires `source_sequences_root` (`prepare_evaluation.py:242–249,297–304`).

   The supplied preparer fixture concretely omits that field (`test_evaluation.py:269–272`) and is nevertheless accepted by the preparation test. For that accepted manifest shape, the runner reaches a `KeyError` after the four shard processes complete, preventing analysis, sealing, and the combined result. Use the existing per-sequence source mapping, or require the additional field before any workload starts. **Whether the actual external base manifest contains this field remains unverified.**

Within the inspected code, the terminal gate checks exit status, matching PID/command receipts, training mode/output, six-pass row accounting, and final hashes. The frozen source comparisons target the training snapshots; I found no basis here to treat those snapshots as mutable. The preparer and evaluator also construct RGB/bbox initialization keys identically.

**Remaining Unverified Interfaces**

- **Final and bank loading:** preparation deserializes protocol metadata and trusts the training roundtrip receipt; actual strict A/B/C loading occurs at `runtime.py:164–165`. Compatibility with the frozen runtime, inherited `FullDenseTracker.initialize`, and real bank tensors was not exercised.
- **OPE:** the locked metric must accept the arguments at `evaluation.py:194–196` and return the count and percentage fields consumed by the evaluator and runner. Its implementation is outside the permitted files. Actual dataset layouts and case contents were likewise not inspected.
- **VOT initialization:** the bridge’s region representation, RGB/depth channel ordering, termination convention, and exact correspondence between toolkit initializations and bank keys remain unverified.
- **VOT outputs and auditing:** toolkit trajectory filenames, analysis CLI/JSON structure and metric units, failure-auditor signature and length checks, and identity-tool arguments/interpreter dependencies remain unverified. The local fixtures establish control-plane expectations, not compatibility with those external implementations.