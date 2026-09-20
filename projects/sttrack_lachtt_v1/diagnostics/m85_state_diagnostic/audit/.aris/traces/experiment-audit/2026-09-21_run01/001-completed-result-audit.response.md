M85 completed-result audit is finished. **Semantic verdict: WARN, same-family/provisional. Deterministic saved-artifact checks: PASS.** No fake GT, primary metric rescaling, missing claimed output or numerical mismatch was found.

- Independently recomputed all 2616 saved positions from NPZ/JSON plus exported dataset GT. Every reported per-frame diagnostic metric, candidate count/index, crop flag, summary and H10 interval matches exactly; maximum reported metric error is 0.0. Selected dense-decode error is at most 3.051757818184342e-05 pixels, within the frozen 1e-4 tolerance.
- Category bbox, score and previous-state rows exactly match all three M84 raw trajectories. The 101-position preflight is an exact prefix of saved rows and dense arrays. All 283 parent manifest files, 161 integrated source hashes and 481 binding checks pass.
- There are 2200 valid GT frames and 646 Category-low frames: 627 have the GT center outside the current crop, 18 have the center inside without a correct Category dense candidate, and 1 has a correct dense candidate. Of the 646 low frames, 645 have no correct candidate in any of the three saved heads. Raw, Swapped and native selected-box rescue counts are all zero; severe-harm counts are also zero under the stated thresholds.
- H10 is identical across Category/Swapped/native under Category history: mobilephone [497,701), car [215,550) and [644,675), ghostmask [847,902). The one candidate-capacity exception is mobilephone frame index 291; none of the selected heads chooses its correct dense candidates.

The WARN preserves two distinctions. Primary IoU/H10/candidate metrics are **real_gt**; auxiliary `native_spatial_kl` uses prediction-sum normalization and a native-response reference, so it is a **self_supervised_proxy**, not accuracy. The original replay's Empty/native full-map equality and noncommit assertions have bound source plus complete receipt/log evidence; all saved Empty/native scalar/box records were independently verified equal, but full Empty maps and query/template snapshots were not exported. I did not rerun GPU inference or training. Native backbone weights, development text-bank bytes and original RGB/depth frames were not independently checked from this local export; the final Category adapter checkpoint was hash-verified.

The evidence remains a three-sequence posthoc state diagnostic. It does not prove earlier template/query causality, independent alternative trajectories, unbiased prevalence, benchmark improvement or M84 promotion. M84's failed gate verdict remains unchanged.

Reports and reproducible evidence:

- `D:\Program Files\UserCache\gb\codex\tmp\sttrack_m85_state_diagnostic_20260921\audit\EXPERIMENT_AUDIT.md`
- `D:\Program Files\UserCache\gb\codex\tmp\sttrack_m85_state_diagnostic_20260921\audit\EXPERIMENT_AUDIT.json`
- `D:\Program Files\UserCache\gb\codex\tmp\sttrack_m85_state_diagnostic_20260921\audit\independent_recompute.py`
- `D:\Program Files\UserCache\gb\codex\tmp\sttrack_m85_state_diagnostic_20260921\audit\independent_recompute.json`
- Full request/response and actual reviewer metadata: `D:\Program Files\UserCache\gb\codex\tmp\sttrack_m85_state_diagnostic_20260921\audit\.aris\traces\experiment-audit\2026-09-21_run01`.

No frozen source or result was modified. All experiment conclusions above were rechecked against current local artifacts; the memory lookup supplied only the pre-existing evidence-bound reporting convention.

<oai-mem-citation>
<citation_entries>
MEMORY.md:411-418|note=[Prior evidence binding and prediction sealing convention rechecked in current source]
</citation_entries>
<rollout_ids>
01a06f16-fd85-7da3-a232-113efe970f88
</rollout_ids>
</oai-mem-citation>