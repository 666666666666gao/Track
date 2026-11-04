**BLOCKING: No blocking findings** at `99dfd9f3f1e49b6684fea4d60b4f5176bdba162a` against the stated training contract.

Read all five requested port files and all seven required legacy sources using `git show HEAD:...`.

- **Gradients and initialization:** A/B/C are freshly initialized after seed2027. AdamW uses LR `3e-5`, weight decay `.01`, and accumulation32, with correct normalization and sequence-tail flushing. C consumes detached features and a detached selected-box IoU target, so its backward pass is independent of A/B. Native and CLIP remain frozen. See [train.py:57](C:/Users/gb/projects/Track_inspection_20261101/projects/sttrack_lachtt_v1/training/cadtrack_innovation_port/train.py:57), [train.py:123](C:/Users/gb/projects/Track_inspection_20261101/projects/sttrack_lachtt_v1/training/cadtrack_innovation_port/train.py:123).

- **Own history, devices, and label timing:** Initialization and observations run on GPU1; initial features, language inputs, and current observations move to GPU0. Subsequent crops use the detached predicted box. Selection and template-write decisions finish before current GT enters either loss. Writes require native cadence5, selected-cell Hann confidence `>.65`, and raw C prediction `>.5`. See [runtime.py:89](C:/Users/gb/projects/Track_inspection_20261101/projects/sttrack_lachtt_v1/training/cadtrack_innovation_port/runtime.py:89).

- **Fixed completion:** The bound sequence list must contain all152 sequences. Production executes six complete passes, totaling **912 sequence runs and 1,318,812 tracking calls**, with an explicit final call-count assertion. No Test50/CDTB/VOT selection or tuning path appears. See [train.py:99](C:/Users/gb/projects/Track_inspection_20261101/projects/sttrack_lachtt_v1/training/cadtrack_innovation_port/train.py:99), [train.py:176](C:/Users/gb/projects/Track_inspection_20261101/projects/sttrack_lachtt_v1/training/cadtrack_innovation_port/train.py:176).

- **Saved model and evaluation loading:** The final contains A/B/C and protocol metadata. The common factory rejects smoke weights, checks native/config/CLIP bindings, strictly loads both learned modules, and freezes them for evaluation. See [runtime.py:151](C:/Users/gb/projects/Track_inspection_20261101/projects/sttrack_lachtt_v1/training/cadtrack_innovation_port/runtime.py:151).

**NONBLOCKING limits**

- All three executable-file hashes match `validation.json`. Those receipts establish only the recorded 64-frame controls; they do not establish six-pass completion or exercise the final factory across all three evaluation datasets. See [validation.json:13](C:/Users/gb/projects/Track_inspection_20261101/projects/sttrack_lachtt_v1/training/cadtrack_innovation_port/validation.json:13).
- External native loader code, checkpoints, data, and launch resources were not independently verified. No fresh runtime tests ran because the local Python launcher exits with `No pyvenv.cfg file`.

No files edited; no SSH or network used.