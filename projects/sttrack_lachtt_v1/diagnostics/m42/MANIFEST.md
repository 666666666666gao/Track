# M42 artifacts

Created2026-09-05:

- `EXPERIMENT_SPEC.md`: scientific scope, model/data/optimization contract.
- `EXPERIMENT_PLAN.md`: execution order and evaluation dependencies.
- `EXPERIMENT_TRACKER.md`: launch and terminal-status ledger.
- `tools/prepare_sttrack_m42.py`: Train-only manifest/label separation.
- `tools/collect_sttrack_m42.py`: causal native-token capture with replay parity.
- `lib/test/tracker/sttrack_local_spatial_observation.py`: ROI/reference bank.
- `lib/models/sttrack/lachtt_local_spatial_association.py`: matched association heads.
- `tools/train_sttrack_m42.py`: frozen fitting and one final development readout.
- `tools/test_sttrack_m42.py`: geometry/model/gradient contracts.
- `tools/run_sttrack_m42.sh`: two collection jobs followed by paired fitting.

Server-only feature tensors, GT labels, traces and large base checkpoint are
not published. Small machine-readable settings and receipts are copied to the
repository diagnostics directory. Runtime source is copied to the repository
overlay and bound by its `MANIFEST.sha256`.
