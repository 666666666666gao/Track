Verdict: **PASS on findings (1), (2), and (4); WARN/qualified PASS on (3)** because the captured source does not include `LTRTrainer` internals. These are **future original-training-entry readiness findings only**. I found no evidence that they explain current M54/native inference behavior, and I am not treating them as causes of any current tracking failure.

1. **Original training paths and configured SOT pretrained file are not ready: PASS.**

   The captured original `local.py` points to stale `/home/hxt` and `/nasdata` paths: `workspace_dir` at `lib/train/admin/local.py:3`, tensorboard/pretrained roots at `local.py:4-5`, and `depthtrack_dir` at `local.py:15`. The timestamped observation records those paths as missing: workspace missing at `observation.json:42-48`, tensorboard missing at `observation.json:49-54`, pretrained networks missing at `observation.json:55-60`, and configured `depthtrack_dir` missing at `observation.json:61-66`. It separately observes the actual local DepthTrack train directory exists at `/root/autodl-tmp/depthtrack/train` at `observation.json:67-72`, which is not the path used by original `local.py`.

   The captured config sets `MODEL.PRETRAIN_FILE: "SOT_Pretrained_256.pth.tar"` at `experiments/sttrack/deep_rgbd_256.yaml:41-43`; the observation records that default file path as nonexistent at `observation.json:73-78`. So the original training entry is not launch-ready without path/config correction.

2. **Changing `MODEL.PRETRAIN_FILE` to `STTrack_Vot22.pth.tar` alone would not load the full native checkpoint in `build_sttrack`: PASS.**

   `build_sttrack` only passes a file to the backbone constructor when `cfg.MODEL.PRETRAIN_FILE` exists, does **not** contain `"STTrack"`, and `training` is true: `lib/models/sttrack/sttrack.py:199-207`. If the string is `STTrack_Vot22.pth.tar`, this branch sets `pretrained = ''`, so the backbone preload path is disabled.

   The only full-model `load_state_dict` in `build_sttrack` is gated on `'SOT' in cfg.MODEL.PRETRAIN_FILE and training` at `lib/models/sttrack/sttrack.py:231-235`. A `STTrack_Vot22.pth.tar` filename would not satisfy that branch either. The original train script creates the network with `net = build_sttrack(cfg)` at `lib/train/train_script.py:50-54`, moves it to CUDA at `train_script.py:56-63`, creates the optimizer at `train_script.py:77-79`, and starts `trainer.train(..., load_latest=True, fail_safe=True)` at `train_script.py:88-89`; it does not explicitly load `STTrack_Vot22.pth.tar` before optimizer construction.

   Correction nuance: with the **captured default** `SOT_Pretrained_256.pth.tar`, `build_sttrack` is designed to call `torch.load(pretrained_file)` and `model.load_state_dict(..., strict=False)` at `sttrack.py:231-235`; the readiness problem there is that the observed default SOT file does not exist. With a hypothetical `STTrack_Vot22.pth.tar` substitution, the readiness problem changes to “not loaded by this training builder.”

3. **EPOCH 15 with VAL_EPOCH_INTERVAL 50 likely means no validation epoch in 1..15, but trainer internals are missing: WARN / qualified PASS.**

   The config sets `TRAIN.EPOCH: 15` at `experiments/sttrack/deep_rgbd_256.yaml:60-62` and `TRAIN.VAL_EPOCH_INTERVAL: 50` at `deep_rgbd_256.yaml:70-74`. The validation loader is constructed with `epoch_interval=cfg.TRAIN.VAL_EPOCH_INTERVAL` at `lib/train/base_functions.py:153-166`. The train script then calls `trainer.train(cfg.TRAIN.EPOCH, load_latest=True, fail_safe=True)` at `lib/train/train_script.py:88-89`.

   This supports the proposed warning under the usual `epoch_interval` semantics: a validation loader scheduled every 50 epochs would not run during epochs 1 through 15. I did **not** find `LTRTrainer` source in the captured bundle or publisher overlay, so I cannot independently prove the trainer’s exact epoch scheduling behavior from primary source. The correct wording is: **the captured loader/config metadata is consistent with no validation during a 15-epoch run, subject to `LTRTrainer` implementation.**

4. **Official 6-sequence validation split overlaps current M54 project usage by 3 fit and 1 development sequence: PASS.**

   The timestamped observation records the official split as 146 train sequences, 6 validation sequences, and 0 train/val overlap at `observation.json:80-92`. The six official validation names are listed at `observation.json:83-90`: `bag04_indoor`, `ball16_indoor`, `bottle03_indoor`, `flower03_indoor`, `pigeon05_wild`, and `toy03_indoor`.

   The same observation records overlap with current M54: fit overlap contains `bag04_indoor`, `bottle03_indoor`, and `toy03_indoor` at `observation.json:93-98`; development overlap contains `flower03_indoor` at `observation.json:99-101`. Current M54 scope is 63 fitting sequences and 22 development sequences at `diagnostics/m54/spec.json:11-12`; the plan also states it uses M44’s 63 fit and 22 development sequences and that the development set has been reused, so it cannot be called a fresh unseen test at `diagnostics/m54/EXPERIMENT_PLAN.md:19`.

   Therefore the official 6-sequence validation split is not a fresh holdout for this project context, even though it is an official validation split internally separated from the original DepthTrack train list.

5. **Separation from current M54/native inference: verified distinction.**

   These findings apply to the **captured original training entry** under `C:/Users/gb/.codex_remote_staging/native_training_entry_20260906`, where `training_launched` is false, `optimizer_steps` is 0, `m54_changed` is false, and no training images/GT were opened at `observation.json:103-107`.

   They do **not** apply to already-audited native inference. The native OPE spec explicitly binds `STTrack_Vot22.pth.tar` as the inference checkpoint at `diagnostics/native_ope/spec.json:4-8`, and the native test tracker loads `self.params.checkpoint` strictly into the network at captured `lib/test/tracker/sttrack.py:20-30`. The native OPE runner passes `spec['checkpoint']` into tracker parameters at `overlay/tools/run_sttrack_native_ope.py:35-39`.

   They also do **not** invalidate the current M54 reader path. M54’s shared parameter builder passes the bound base checkpoint into tracker params at `overlay/tools/sttrack_m54_common.py:32-36`; M54 recursive execution loads the trained reader checkpoint, verifies its `spec_sha256` and `base_checkpoint_sha256`, and strictly loads the reader weights at `overlay/tools/run_sttrack_m54.py:91-95`. The M54 training script trains only `TemplateReader()` at `overlay/tools/train_sttrack_m54.py:70-73`, after sealed feature/GT ordering checks at `train_sttrack_m54.py:23-65`.

Remaining limitations:

- I did not launch training, test path existence beyond the timestamped observation, contact the server, inspect live remote state, use GPU, or run new tests.
- I did not inspect `LTRTrainer` internals because they were not present in the captured source or publisher overlay found in this bounded read. The validation-interval finding should remain phrased as loader/config-readiness evidence, not a fully proven trainer-behavior claim.
- I found no concrete contradiction to the proposed readiness findings.