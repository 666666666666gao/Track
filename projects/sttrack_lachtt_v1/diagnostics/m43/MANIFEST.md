# M43 files

- `EXPERIMENT_SPEC.md`: new performance question and explicit M42 failure status.
- `EXPERIMENT_PLAN.md`: ordered execution and public-evaluation condition.
- `spec.json`: source, checkpoint, dataset and comparator identity.
- `tools/prepare_sttrack_m43.py`: freezes the run before new recursive outputs.
- `tools/run_sttrack_m43.py`: two independent inference arms and posthoc analysis.

Server-only trajectories remain under `trajectories/`. Publish final small
results and receipts after completion. No model weight or dataset payload is
added to GitHub.
