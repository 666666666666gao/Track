# M125-R0 Full Result Experiment Audit

Date: 2026-10-18
Overall verdict: WARN
Source verdict: PASS_SOURCE_ONLY
Result verdict: RESULT_INTEGRITY_PASS_WITH_WARNINGS

The listed observations establish a completed fixed-six-pass controller and a complete official DT50/CDTB80/VOT127 evaluation. All nine reported values exist in embedded result files and bind to one pass-six final and one bundle. WARN reflects five failed target gates, explicit fresh_result_audit_complete=false / independent_completed_audit=false, unavailable raw payloads for independent recomputation, one-seed/disclosed scope, and UNATTESTED actual routing.

## Exact nine metrics and target comparison

| Dataset | Metric | Exact value | Target | Result |
|---|---:|---:|---:|:---:|
| DepthTrack Test50 | Precision | 70.20966734349989 | >= 65.2 | PASS |
| DepthTrack Test50 | Recall | 61.50369567358845 | >= 64.9 | FAIL |
| DepthTrack Test50 | F-score | 65.56895845227588 | >= 65.1 | PASS |
| CDTB80 | Precision | 75.45961789325742 | >= 72.9 | PASS |
| CDTB80 | Recall | 66.17577807419001 | >= 75.6 | FAIL |
| CDTB80 | F-score | 70.51343194486606 | >= 74.2 | FAIL |
| VOT127 | EAO | 75.78624268877728 | > 77.9 | FAIL |
| VOT127 | ACC | 82.31854450835942 | > 82.1 | PASS |
| VOT127 | ROB | 91.35982054000516 | > 93.7 | FAIL |

Four of nine gates pass; five fail. any_joint_pass=false and joint_pass=false.

## Completed-suite evidence

Primary observation actual_original_M125_R0_six_pass_end_node_observation at 2026-10-18T04:58:21.565663+00:00 contains 79 embedded files. Every declared byte count and SHA-256 recomputes correctly. R2 observation actual_original_R0_final_metric_files_read_only_observation at 2026-10-18T05:03:47.765280+00:00 contains the six final metric/result files; all six recompute correctly, and its three overlapping result/selection texts are byte-identical to the primary observation.

Controller terminal exit is 0, with zero progress queries and zero automatic retries. Primary observation records NN calls 0 and GPU queries 0; R2 records NN calls 0, GPU queries 0, training-progress queries 0, and automatic retries 0.

All six pass receipts and nine post-training receipts (15 total) exit zero:

01. train_fixed_pass (pass 1) — exit 0, receipt SHA cbacb125d776b5d567ab9ec48cc6c19d552c11f880b456eb9f59f7ced978f374
02. train_fixed_pass (pass 2) — exit 0, receipt SHA 7556da9380b9a45aaa8c1ffed3ef3827b6f1aa1c2b313bc1fe9a0ec0c76e4513
03. train_fixed_pass (pass 3) — exit 0, receipt SHA 4cb795b78abc36efdde792af5da36f90ad7d95cec92cf25fafb8af468063cf14
04. train_fixed_pass (pass 4) — exit 0, receipt SHA 49b6a6691c55efcf96c9f9692bad4f12e2f0d1ca3e8aa5b3fb2f627bbab9cab2
05. train_fixed_pass (pass 5) — exit 0, receipt SHA eaa5c5b024b21f9fcb4ca872b97db66e73ceb9dc19f08b47cbae41d2146c5b36
06. train_fixed_pass (pass 6) — exit 0, receipt SHA 8c835b04ca912d412f8de09675c5e2931e8f7bc36f58d4f979e107de78da412d
07. prepare — exit 0, receipt SHA 7d581a6e0403a9cdbbda8742a7911d6f9eb58b10913c58c7dd3769b9ac64de56
08. depthtrack_test_track — exit 0, receipt SHA 5a6aab91380bebefe89f6c6ea001b118a8687799e12c9be11816e32314231815
09. cdtb_track — exit 0, receipt SHA 18c538c5bbea204201a8fc045a53bc4be204a9cb3dfcf6ec398c6b0010aa4200
10. depthtrack_test_analyze — exit 0, receipt SHA fed4e8d9d38738d7880cdf6a90480716fe93ee4d12c7ac1a5273043c8b8222a5
11. cdtb_analyze — exit 0, receipt SHA 15caae6d8edccddfe4f0052aba3f8efdc3f657db07283d60b1cfbb2b10607c8c
12. vot_track — exit 0, receipt SHA 74e8482cbcd92b5314814f6587c174df7dfe81f4b5ff956e56c077118407bddd
13. vot_official_analysis — exit 0, receipt SHA 6f43f00e54721f64c0bedc6f136d42043947e3534d085bd59201f2e87cec7adb
14. vot_seal — exit 0, receipt SHA 78012a044662de7ea30cc6e71b690da68e4dfaad554a82cd2fe5650a181c5a06
15. collect — exit 0, receipt SHA 0295b0910bf35c664f73bc0f141e757b632a276a0d2c27ef31ef9efd5f9eb050

Pass summaries:

- Pass 1: epoch 1, runs 152, calls 219802, optimizer 6807, final SHA 739585378c02846909b0f86bc676d27be82aa3f586bdf527c1efd6ead8bf1195, resumed None.
- Pass 2: epoch 2, runs 304, calls 439604, optimizer 13614, final SHA 32de97f200b678d349efd40928bc8b1197761d98c2440fa18cb009b058e10976, resumed e191562ded4ba80953aea8d77b6aba9aace16ece2458ce8d33c0f82056a8f37f.
- Pass 3: epoch 3, runs 456, calls 659406, optimizer 20421, final SHA 258d6316a769442d864670e012a7f4fb473cb7235a43873b36cbd08cd04d8a06, resumed cd241e73494d60bf6870d1453c961de8e912d973ad595a19c9d25388a60fb08b.
- Pass 4: epoch 4, runs 608, calls 879208, optimizer 27228, final SHA fa849fd18e149921d6a67fd7c33fef2e8477a91a01dc9b83b771c5e9e3c10ec3, resumed 5652ef4e0ea852f1c16bcb31baae7da6565a732eb5eebb954d12a1720f9f2245.
- Pass 5: epoch 5, runs 760, calls 1099010, optimizer 34035, final SHA c334ae53a9c1114a78a4b5fad9b2c52992237e3d3b71372dd9703543bf57e85d, resumed 242b76d26283ff392d1bf681f40add0e8447c8a15a49016eb20beb9714123b3b.
- Pass 6: epoch 6, runs 912, calls 1318812, optimizer 40842, final SHA cf88e9c345c393caf09ad42fbe21b1bcd3b6c6bacd13bfc5124b12e511bc2a2f, resumed d2470034407dddd6382186f09d30a76728ec5dd9b81bea88e7c47e4cd1d167a1.

Pass six has 912 sequence runs, 1,318,812 track calls, 40,842 optimizer steps, fixed_final_after_complete_budget=true, and final SHA cf88e9c345c393caf09ad42fbe21b1bcd3b6c6bacd13bfc5124b12e511bc2a2f. current_training_state.json states final_weight_chosen_by="predeclared pass six; no external score selection".
The launch record binds gate SHA 5dd84fe5564a493dbb6e7dea334e9001e521494df1a022d8a6c2d01cd06ffcba, source-review SHA 41d75e53380e9fe227bccafb2e29fe6a70e04de51eeab829fc8008f9dd83d762, and real64-result SHA 615268deffcc57afed3a63c3b3af238cb228b9fde3736e29124560f3e83f3019.

## Ground-truth provenance and official metric computation

Static classification is real_gt. run_m125_R0_fixed_official.py:46-61 hashes dataset groundtruth.txt after predictions are sealed and invokes the locked evaluator; :65-80 uses the VOT bridge without inference-time GT. The actual DT50/CDTB80 receipts report subsequent_GT_opened=false, optimizer_steps=0, and frozen-state exactness. VOT has four successful shard launches (441/441/441/442 anchors), a complete 1,765-anchor / 5,295-result-file merge, official analysis exit 0, seal exit 0, and collect exit 0. The raw audit helper lists GT metadata for 130 OPE and 127 VOT entries, but raw M125 GT/prediction bodies are not in the supplied snapshots.

The prior static source review reported 341 bounded deterministic checks with zero failures and PASS_SOURCE_ONLY; that is source/control-flow evidence, not NN execution or independent metric recomputation.

## Source/result binding and same-final proof

selection.json, the all_results.json model/rows, DT metrics, CDTB metrics, and VOT result all carry final SHA cf88e9c345c393caf09ad42fbe21b1bcd3b6c6bacd13bfc5124b12e511bc2a2f and bundle SHA 7178b52586c7ab85630d99cf24187fdf5a2d6d5202bd02b35877094e20172dff. selection.training_result_sha256 is d3932afaa6b0d23f1964dfd5830d3d7885837063b2444b5ecf5d9018b90d7a68, matching the embedded pass-06 fit-result file hash. DT/CDTB receipts bind plan/final/bank/bundle/counts/receipt hashes; VOT binds bank/binding, merge SHA, analysis SHA, 1,765 anchors and 127 sequences. Every result row SHA equals the R2 embedded file SHA; DT/CDTB stored percentages equal raw fractions times 100. This is hash/field equality in the supplied snapshot only; hidden tensor equality is not inferred.

The fixed source runs all six passes before evaluation (run_m125_R0_fixed_suite.py:50-67, :70-99). All pass results and selection/all-results flags mark external metric checkpoint selection and Test50/CDTB80 checkpoint selection false. No hidden post-hoc selection, duplicate, or spliced-metric evidence appears in the listed source/snapshots; this is evidence-bound.

## Scope and limitations

Scope is one seed (2027), six passes over 152 training sequences, DT50 (76,373 frames), CDTB80 (101,956 frames), and VOT127 (1,765 anchors). Historical Test/CDTB engineering exposure and multiframe human-review aids are explicitly disclosed. Result files set fresh_result_audit_complete=false and independent_completed_audit=false; raw trajectories and raw OPE/GT bodies are absent for fresh recomputation. The due-node progress file is pre-completion provenance only.

Requested reviewer route is gpt-6-astra/max, but actual model/reasoning route is UNATTESTED; review is same-family and provisional.

## Input hashes and preservation

Primary observation: edcfcadeb24fa5d30045bf42e956ff5a3971c887405d4889454b3da4b3206847 (3831330 bytes).
Due progress: 4c9d4b1bfa8154dd7dde9aa2a1c17c9fdc2d1eb178190a6038c116b5c79527d0 (4760 bytes).
Raw helper: ba7fedad376a09cd136302adfe6448d5e047a3bbc778b727c8ec50cc9e52916b (27355653 bytes).
R2 final files observation: 7616215ca7561a21342d2f3a85cd33daa74cbf914bb3f9a68ada076791b78c36 (530927 bytes).

All 55 staged source paths covered by the prior source audit are unchanged by SHA-256; the prior audit inventory contained 73 inputs in total. The current staged source directory has 56 files; its extra CANDIDATE_STORAGE_R2.json is recorded in the JSON inventory. The documented EXPERIMENT_PLAN.md append was already reviewed as DOCUMENT_APPEND_REVIEWED_SOURCE_BYTES_UNCHANGED. No input was modified; only the two requested report files were created.

## Skill checks and claim impact

- Ground truth: PASS, real dataset GT bindings; raw bodies unavailable for independent recomputation.
- Score normalization: PASS (static operator review); no model-output denominator.
- Result existence/numbers: PASS in supplied snapshots; hashes, keys, and exact values agree.
- Dead code/reachability: PASS (static plus 15 actual receipts, all exit zero).
- Scope: WARN (one seed and disclosed historical exposure/review aids).
- Evaluation type: PASS, real_gt for DT/CDTB/VOT.

Completed controller and nine metric file existence are supported. Joint target attainment is unsupported because five gates fail. Clean held-out or multi-seed claims need qualification. Independent raw metric reproduction is not established.

See EXPERIMENT_AUDIT.json for every input hash, receipt hash, pass summary, binding check, route attribution, and target row.
