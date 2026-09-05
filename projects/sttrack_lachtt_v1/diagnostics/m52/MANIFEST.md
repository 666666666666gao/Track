# M52 artifacts

- spec.json and EXPERIMENT_PLAN.md freeze the policy-state data and paired training design.
- contract.json / contract.exit: observed vs plain M45 equality for 240 frames; PASS.
- run.sh / execution_binding.json / launch.json: actual GPU0 collection job.
- LAUNCH_REPORT.md: data collection RUNNING; no M52 training or performance result yet.
- Large features and full traces stay in the remote run root, with hashes in completion receipts.
- train_sttrack_m52.py / run_sttrack_m52.py are in the overlay; training_binding.json,
  run_training.sh, and check_runtime.py bind the prepared paired pipeline.
- PRE_TRAINING_NOTE.md freezes data-effect versus extra-training advancement before training.
- PIPELINE_IMPLEMENTATION.md records implementation and the limited checks already run.
- EXPERIMENT_AUDIT.md / .json preserve the pre-training Type-A advisory WARN review.
- source_snapshots/sttrack.py supplies the exact native tracker source for review;
  its README records origin, byte count, and the existing source binding.
