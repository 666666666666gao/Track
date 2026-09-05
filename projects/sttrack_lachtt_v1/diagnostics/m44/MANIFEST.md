# M44 artifact roles

- `spec.json`: frozen architecture, sampling, source/base/data hashes and gates.
- `inference_inputs.json`: normal initialization, requested frame pairs and
  expected default predictions; no subsequent GT boxes.
- `training_labels.json`: Train-only target supervision, separate from inference.
- `features/`: server-only raw native paired RoIs; not committed to GitHub.
- `*_receipt.json`: collection completion and source/artifact integrity.
- Final checkpoints: server-only new association weights, requiring the pinned
  official STTrack base. No full-benchmark claim at collection launch.
