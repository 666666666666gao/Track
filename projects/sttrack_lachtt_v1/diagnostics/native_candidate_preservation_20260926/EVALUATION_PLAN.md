# M89 paired final evaluation

Status: protocol prepared before training completion; evaluation not launched.

Both arms use seed2027, all 152 DepthTrack Train sequences, 219802 tracking calls and 6807 optimizer updates. Control uses candidate preservation weight0; Candidate uses weight1. Each arm must finish the fixed full pass, retain frozen base parameters/buffers, and produce a matching final checkpoint and completion receipt. Never select intermediate checkpoints using external metrics.

Reuse the sealed M82-Full152 evaluation interface, Category banks, cases and metric implementation identified in `evaluation_inputs.json`. This reuses inputs only; every prediction must be generated afresh. No caption regeneration, attribute activation, Hann changes or template changes. The new training loss adds no inference module.

| Dataset | Coverage | Metrics |
|---|---|---|
| DepthTrack Test | 50 sequences, 76373 frames | P / R / F |
| CDTB | 80 sequences, 101956 frames | P / R / F |
| VOT-RGBD2022 | 127 sequences, 1765 anchors | EAO / ACC / ROB |

Run Control followed by Candidate. Within each arm, run DepthTrack on GPU0 and CDTB on GPU1 concurrently, then VOT using the established two-GPU shards. Training owns both GPUs until its completion is verified. Use new output directories and tracker identities; do not copy old predictions.

Before binding final weights, verify the M89 experiment spec, source hashes, arm weight, common initialization, training counts and checkpoint/result agreement. Rebind bundle/checkpoint hashes and output paths while preserving all frozen evaluation inputs. Verify all bank, case and metric hashes against the actual files, not only their plan entries.

After inference, verify exact dataset coverage, receipt hashes and VOT merge coverage before computing metrics. Report all eighteen main metrics, per-sequence gains and harms, and confirmed failed anchors. A single model must satisfy all project targets; do not combine best entries across arms or datasets. Training completion is not performance acceptance.

`prepare_evaluation.py audit-inputs` has passed against the actual remote files. `prepare_evaluation.py bind` validates both completed final checkpoints and creates fresh bundles and OPE/VOT plans; its final-binding path cannot yet run because training is ongoing. Remaining implementation: VOT workspace generation and completion-controlled evaluation launch. No evaluation process has been started by preparing this plan.
