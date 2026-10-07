# M122: full127 ROB contribution accounting

This independent, CPU-only diagnostic runs **after** the two fixed finals have completed all six official evaluations. It does not modify the running 32-source evaluation gate, launch neural jobs, update model parameters, choose checkpoints, tune thresholds, or change tracking outputs.

## Inputs and scope

- Require the existing `all_results.json` six-evaluation seal and its exact `selection.json` digest.
- Read the two sealed full127 results, final/bundle/text-bank/initialization-binding digests, and the 5,295 merged result-file digests per model.
- Compare with the already sealed native full127 result. Require the same 1,765 anchor keys, run lengths and failure protocol.
- Verify the installed official multistart source digest and reconstruct each recorded ROB to `1e-10`.
- Keep EAO and ACC as official aggregate results. This tool does not claim additive contributions for those two metrics.

For each sequence, official ROB weights its aggregate survived anchor-frame fraction by the sequence length. A sequence contribution is its weighted model-minus-native ROB difference. An anchor contribution uses the same sequence weight and denominator, and the model-minus-native survived-frame difference. Both sums must reproduce the exact aggregate ROB difference.

Outputs retain all 127 sequence rows and all 1,765 anchor rows for each final, including improvements and harms, newly failed and rescued anchors. The top-ten lists are descriptive summaries of the full ledger. Failure count alone does not determine ROB: when the failure occurs also matters.

## Validation and limitations

The pure accounting function is checked locally against the archived native, M67-Full152 and M82-Full152 outcomes. This is an actual historical data check, **not** a new M122 metric, an installed-toolkit CLI execution, or a deployed improvement.

The standalone source must pass a fresh Phase 2.5 source review before remote deployment. Its CLI and the actual M122 accounting remain pending until the six-evaluation seal exists. The review does not establish experimental results.

Accounting identifies which sequences and anchors numerically contribute to a ROB loss or gain. It does not establish a causal explanation such as wrong text, template contamination, candidate ranking, or crop exclusion. Those mechanisms require subsequent targeted replay of real saved states. In particular, six-decimal saved boxes cannot establish exact historical crop rounding at a boundary.

Example, after the suite is sealed:

```sh
/root/miniconda3/envs/mplt/bin/python analyze_m122_rob_contributions.py \
  --suite /root/autodl-tmp/sttrack_m122_official_evaluation_20261007 \
  --output /root/autodl-tmp/sttrack_m122_rob_accounting_20261007
```

No new M122 scores exist at preparation time. The original hourly observer and two-GPU full training continue unchanged.
