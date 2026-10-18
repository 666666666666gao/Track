# M125-R0 fixed-six final result

This is the first completed M125-R0 candidate under the unified protocol: only DepthTrack Train152 was used for training; one final checkpoint and one primary inference setup were evaluated on all three benchmarks. All six training passes and the full DT50/CDTB80/VOT127 suite completed with zero stage exits and no automatic retries.

| Dataset | Metrics | Target comparison |
|---|---|---|
| DepthTrack Test50 | P 70.209667 / R 61.503696 / F 65.568958 | P/F pass; R fails 64.9 |
| CDTB80 | P 75.459618 / R 66.175778 / F 70.513432 | P passes; R/F fail |
| VOT-RGBD2022 | EAO 75.786243 / ACC 82.318545 / ROB 91.359821 | ACC passes; EAO/ROB fail |

The same final SHA is `cf88e9c345c393caf09ad42fbe21b1bcd3b6c6bacd13bfc5124b12e511bc2a2f` in all rows. The VOT run has 1,765 anchors and 5,295 trajectory files; confirmed failures are 249. Exact metric-file SHA values are checked against `evaluation/all_results.json` in the metric observation and result audit.

This R0 result is a completed baseline and does not meet the full target. It does not authorize per-dataset checkpoint selection. The next experiment is the already-reviewed uniform CAD U0/U1/U2 inference comparison on this fixed R0 final, subject to the source gate and actual resource admission; ROI training remains deferred until after CAD.

The independent result audit is recorded beside this README. Its reviewer is same-family/provisional and actual model routing is UNATTESTED; the audit is evidence about integrity, not a claim of cross-family certification. Historical M124 test-informed selection and manual text-input boundaries remain disclosed in the main document.
