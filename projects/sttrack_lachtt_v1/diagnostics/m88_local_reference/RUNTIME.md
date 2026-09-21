# M88 runtime and evidence locations

Reuse the installed STTrack environment `/root/autodl-tmp/envs/sttrack/bin/python` (Python3.8 / torch1.13.1cu116); no package or driver changes. Two NVIDIA3090 GPUs were observed idle before preparation. Dataset/base/CLIP/Qwen files remain at existing paths; banks reuse M84's frozen paths. Runtime root `/root/autodl-tmp/sttrack_m88_local_reference_20260921`.

Source preparation and CPU model check completed before any tracker calls. `preparation_receipt.json` preserves the first successful preparation bindings. `numerical_clarification.json` records the later pre-freeze plan wording adjustment and final spec identities. No model/performance gate changed with that clarification.

GPU preflight command (after fresh code review):

```bash
cd /root/autodl-tmp/sttrack_m88_local_reference_20260921
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 /root/autodl-tmp/envs/sttrack/bin/python -u /root/autodl-tmp/sttrack_m88_local_reference_20260921/preflight_m88.py
```

The prefix and96-frame optimization weights are discarded. Formal training starts only from the saved M84-identical zero parameter tensors under the new architecture tag. `freeze_m88.py` checks reviewed sources/spec/preflight before `launch_m88.py` starts a detached process. `run_m88.sh` stores each exit code and stops on errors. It runs Category training, Category/Empty recursions on GPU0/1, Swapped onGPU0, then all-three-sealed GT analysis. No formal external evaluation is automatically launched. Follow actual PIDs; check roughly every240 seconds near expected milestones, not intermediate-loss tuning.

Preparation failures are preserved: initial `prepare.py` expected only bank entries but encountered existing integer metadata; corrected by naming actual arms. Synthetic search permutation bitwise test and Empty gradient bitwise test were replaced by explicit measured floating tolerances; identical originalM84 Empty test has the same2.384e-7 remainder. Exact Empty inference output remains required.

The first GPU preflight passed in32.110 seconds, but freeze's `bash -n` then rejected CRLF line endings in the queue. No frozen.json or full training was created. The queue was converted to LF only, syntax passed, specs/review were rebound, and the same short preflight was repeated under that final binding. Prior successful preflight and specs remain in `preflight_attempt1/`; this duplicate preflight is not an independent performance experiment and both discarded passes must be counted in preparation cost.

After completion, collect final checkpoint, all130 sequence logs, sampled state trace, all66 prediction files, receipts/specs/source/banks and22 developmentGT files in a private archive. Independently recompute metrics,14 gates, full Empty bbox/score equality, and strict damage/improvement. Public source and numeric reports omit GT coordinate files, weights and images. Three handoff copies remain canonical repo docs, Windows Desktop document, remote project docs.
