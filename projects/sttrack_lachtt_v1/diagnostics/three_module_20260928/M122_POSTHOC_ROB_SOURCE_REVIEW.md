# M122 post-seal ROB source review

**PASS — source only; no blocking findings.** Fresh reviewer `/root/m122_posthoc_rob_source_review`; `review_independence: same-family`, `acceptance_status: provisional`. Reviewed at 2026-10-07 07:15:58 UTC. The workflow requests GPT-6-Astra/max; this reviewer cannot independently attest backend model/effort metadata.

Reviewed source: `analyze_m122_rob_contributions.py`, SHA256 `63398f36955a1b936ba05e20c75807f40de3af6a242cd4ac83bd1a4cd0ca4e6e`. No implementation changes are required.

- The formula matches the historical full127 analysis: official ROB is 100 times the sequence-length-weighted mean of each sequence's summed survived anchor frames divided by summed available anchor frames. The denominator is 80,741 sequence frames. All 127 sequence and 1,765 anchor rows per final are retained.
- Both sequence and anchor contributions sum to the model-minus-native ROB difference within `1e-10`. The reviewer also independently checked exact rational arithmetic and every sequence's anchor subtotal on actual archived native/M67/M82 outcomes.
- The new CLI requires the existing six-evaluation seal and exact selection digest, then follows the pinned `precision0` and `precision1` final/bundle/bank/binding chain. Paths and keys match the four current producers. It checks the two VOT result digests, official analysis digests, all 5,295 merged-file digests per final and actual final bytes. The existing preparation binds the two third-pass finals to 50 + 80 + 1,765 = 1,895 human initialization inputs.
- Python 3.8 AST parsing passes for the new analyzer and four integration sources. The actual local accounting check imports no Torch or VOT toolkit. The standalone analyzer performs no neural execution, parameter update, checkpoint selection, tracking-output change or queue mutation.
- All 32 original evaluation-gate source hashes still match the prior review receipt. The running gate was not edited.

The historical data check reproduced the following values; these are **not M122 scores**:

| Archived model | Reconstructed ROB (%) | Difference from native (pp) |
|---|---:|---:|
| Native | 93.6691093676817 | — |
| M67 | 92.10679490126314 | -1.562314466418556 |
| M82 | 90.23832654452518 | -3.4307828231565196 |

**Nonblocking limitations.** The real M122 six-evaluation seal and scores are pending; no M122 accounting or completed-result audit was performed. Local execution used Python 3.13 stdlib with a Python 3.8 syntax check. The installed Python 3.8 toolkit import/digest and complete CLI remain to be verified after the seal exists. This diagnostic trusts the canonical collector's complete seal for the four OPE evaluations and does not independently re-audit human labels or recreate the text bank.

The output is an additive ROB ledger against native. It cannot establish EAO/ACC contributions, deployed improvement, a causal precision-arm effect or a failure mechanism. The plan correctly preserves these limits, including the insufficiency of six-decimal saved boxes for exact historical crop rounding.

Reviewed source hashes:

| File | SHA256 |
|---|---|
| `projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/analyze_m122_rob_contributions.py` | `63398f36955a1b936ba05e20c75807f40de3af6a242cd4ac83bd1a4cd0ca4e6e` |
| `projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/collect_m122_official_results.py` | `5bcb7f0316f5fcab0205c0ca061152f7a47e3cc6bb57433b50bf7eca5ce555c6` |
| `projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/analyze_m122_vot.py` | `97d90420c63f7785be28c144652dde2bc25ca9dd5c28c70bf245ba2b45abc5d0` |
| `projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/run_m122_vot_shards.py` | `9c2b55bb77980bba5ec486b1ff171a1a1c4de31725167f49ad0f411345a365e8` |
| `projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/prepare_m122_evaluation_suite.py` | `1428b3488d37cbbb71ae66eeca896b553c14d4156c9ab65657c59f741d321cea` |
| `projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/M122_POSTHOC_ROB_PLAN.md` | `f75ebf687b2f9d0551532a4df6a0d8b00169622ea8aa614eefd87a2c22d6d7a1` |
| `projects/sttrack_lachtt_v1/diagnostics/full152_paired_20260925/completed/analyze_vot_rob_contributions.py` | `0fbda0cd61cb354d9b4c9cda8823a88d873a67f89d936505cad0ac46cccb97ea` |

The full path hashes, actual local check outputs and explicit `blocking_findings: []` are recorded in `.aris/m122_full_causal_20261007/posthoc_rob_source_review_20261007.json`. The supplied historical check receipt `ROB_archived_function_check_20261007_1508.json` matches this source and all three input hashes. This review made zero remote connections, GPU/job queries, package installs, neural executions or optimizer steps. Prior reviews and training files were preserved.
