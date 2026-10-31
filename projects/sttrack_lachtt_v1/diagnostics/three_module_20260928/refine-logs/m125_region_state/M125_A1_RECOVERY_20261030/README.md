# A1 recovery and ROI launch, 2026-10-30 New York

This directory preserves outputs copied from the authenticated AutoDL instance.
It contains no weights, datasets, credentials, or full trajectory payloads.
The original evidence archive SHA256 is
`df2447e0fc89fa94ba758cd39aa48187daf96845fbf35fdc3c047654f44fbcad`.

- `m125_A1_recovery_20261030/recovery_result.json`: complete OPE recomputation, input/source/trajectory hashes, 1765 event bindings, and failure accounting.
- `m125_A1_recovery_20261030/fresh_analysis_check.json` and `fresh_vot_analysis/analysis/`: subsequent fresh official VOT analysis. Its results structure exactly matches the original; whole JSON files differ in metadata.
- `m125_A1_recovery_20261030/ROB_vs_U2.json`, `ROB_vs_native.json`, and CSV: paired anchor outcomes; these do not establish a causal selection mechanism.
- `m125_A1_anchored_U2_m025_20261020_r2/`: original analysis, OPE metrics, executed A1 scorer/wrappers, and saved logs.
- `m125_region_roi_fixed6/`: reviewed source/input gate, real64 witness, and actual six-pass launch receipts. This is a running experiment, not a completed result.
- `m125_A1_recovery_20261030/ROI_launch_observation.json`: process/GPU evidence and the first two completed training sequences.
- `m125_A1_recovery_20261030/fresh_vot_analysis.log`: original successful official-analysis log, recovered in a later SSH copy; its bytes are preserved.
- `roi_health_20261031T015115Z.json`: two observations 240 seconds apart confirming the same supervisor and continued training, with 19/152 sequences completed in pass one.

A1 passes four of nine gates: DT P/F, CDTB P, VOT ACC. Its EAO/ACC/ROB
are 77.22474503871115/82.11693020864159/92.88857127537614 percent.
The existing goal remains unmet. The ordinary ROI arm keeps the fixed
six-pass Train152 budget and R0 inference profile, with no inter-pass
Test50/CDTB80 checkpoint selection.

See the project master document section 5.560 for protocol and limitations,
including historical A1 OPE metadata without checkpoint/bundle hashes.
