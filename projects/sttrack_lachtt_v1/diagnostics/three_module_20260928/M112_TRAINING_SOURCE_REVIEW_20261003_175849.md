# M112 training source review — round 1

**WARN, no blocking source findings.** Actual report time: 2026-10-03T17:58:49.826031+08:00. Reviewer model: gpt-6-astra; reasoning: max; review_independence: same-family; acceptance_status: provisional.

This fresh review read the supplied source files directly and used local CPU standard-library checks. **No SSH, deployment, PyTorch forward, GPU sanity, optimizer update or training was executed.** Runtime gates remain required. The trainer snapshot reviewed in this round is `c94ebf3227cb0fdb4843f5ec05bc70a74910bf750ff17b3b73847bc1621998e5`. It was read before the executor's reporting fix; its bytes were subsequently verified from commit `b7f01434253ebf9ac4adc6fa11ac3b19e7f39d7f`. A final pre-write digest check detected the concurrent change and prevented silently assigning this finding to the new source.

## Nonblocking finding

NB1 (old trainer lines 117–133 and shared summarize lines 114–128): only weak_text/all has an explicit paired own-Empty summary. Generic and healthy/transition summaries are present, but their breaks/rescues compare to native candidate 0, not Empty. This matters on existing data: the completed M111 control generic/transition rows have one native-relative break and zero own-Empty breaks. Add or collect generic and weak_text own-Empty paired summaries for all/healthy/transition from the saved aligned rows. No model/loss/data or training rerun is needed. All raw selection rows needed for this are already saved.

## Source correctness

1. **Labels, split and indices:** the entire private manifest was parsed and checked against private weak_labels. All 202 entries are fit, covering 24 events and 48 real candidates. Counts are 98 supported, 15 conflicting, 89 unknown; 101 phrase pairs include only 10 S/C contrasts. Every row matches the original event, actual A/B cached candidate index and original zero-based phrase slot/query. Candidate indices include 0/1/2/4/6/8, so ordinal A/B indexing would be wrong; the trainer uses actual indices. The fixed manifest digest matches. Semantic development labels remain zero.
2. **Objective and sampling:** current logits are selected as [row, actual candidate, actual slot]. A separate CPU generator seeded 32027+step samples 32 rows uniformly with replacement. CE is unweighted at coefficient 1. Selection BCE against actual candidate IoU and native preservation remain unchanged; old weak identity rank and M111 category supervision are absent. IoU does not manufacture phrase labels.
3. **Budget and initialization:** the common M101 parent/M108 bank and completed zero-control identifiers agree with historical records. Seed 2027, 12 epochs, 480 updates, batch 64 (last batch 48), AdamW 3e-4 and zeroed semantic/phrase readouts match the prior comparison. There is no zero-arm retraining or checkpoint selection on development performance.
4. **Trainable parameters and dtype:** declared dimensions sum to exactly 95,683: text 49,344; slots 320; phrase_read 16,640; text_read 16,640; evidence MLP 8,451; semantic 4,096; phrase 192. Visual parameters are excluded from the optimizer. Shared batch conversion keeps masks Boolean and converts non-Boolean cached tensors to FP32. The contiguous text correction used by completed M111 remains present.
5. **Gradients, frozen state and Empty:** the source rejects nonfinite losses/gradients and any frozen-parameter gradient, checks a direct evidence-head gradient on sanity step 1 and later text/semantic gradients, compares every frozen parameter and buffer exactly, and compares all 3039 before/after Empty score/quality rows. Empty selection must also equal visual selection. These are reviewed assertions, not runtime observations from this review.
6. **Reference equivalence:** the readonly branch loads the fixed completed control, disables all gradients, creates no optimizer and saves no checkpoint. It evaluates the new 202 labels and recomputes all stored fit/development selection rows. Both summaries and all four stored JSONL byte streams must match. The local M111 result records the same final weight digest as M110, and all four M111/M110 row files are byte-identical (2544 fitting; 495 each Empty/generic/weak_text). The actual checkpoint tensor files were not locally read; remote bytes and neural reproduction are still required.
7. **Controller failures:** GPU0 sanity and GPU1 reference both finish before acceptance. Nonzero exits, wrong step/status, missing reference reproduction or failed frozen/Empty flags stop before full training. A training nonzero exit or wrong 480-step budget cannot produce controller success. There is no automatic retry or sibling restart.
8. **Readouts and protocol:** initial/final phrase logits, CE/confusion/recalls/majority baseline and ten support margins are saved. Replacing current region/context/support/depth-valid fields with immutable initial fields is a read-only artificial intervention; text and initialization remain fixed. It is not deployment truth or transfer evidence. Selection is evaluated against actual cached Train GT via IoU, with all condition rows and healthy/transition summaries retained. There is no recursive update, public evaluation, physical-identity truth or automatic promotion.

## Checks actually executed

- Parsed Python syntax for all listed sources and all three embedded remote payloads in the private transport scripts.
- Parsed every private-manifest row and checked event/candidate/slot/query/split/status/count/pair consistency; no private rows or evidence notes are reproduced in this report.
- Compared historical M111/M110 metadata and byte-compared all four stored selection JSONL files.
- Ran nine in-memory mocks using the actual controller source: successful ordering; sanity nonzero; reference nonzero; wrong sanity steps; wrong reference steps; reference reproduction false; reference frozen false; training nonzero; training 479 steps. All passed; zero real child processes were launched.
- Recomputed the trainable count from declared layer dimensions. No PyTorch/model forward or gradient test ran.

The default local python command failed with `No pyvenv.cfg file`. The already installed uv-managed Python 3.13 completed the standard-library checks; torch and numpy are absent there. No environment was installed.

## Private deployment/launch review

Both private scripts were read in full, and their embedded Python was syntax-checked without execution. They use the fixed endpoint and stdin password, and do not record credentials. Deployment checks the review verdict/source bytes, existing remote dependency bytes, local/Desktop master equality, remote normalized master prefix and exact rewritten bytes. It requires two idle GPUs, more than 20 MiB free data space, absent experiment output and a fresh staging directory. It uploads byte-exact sources and a 0600 private manifest, then requests only import preflight with CUDA hidden. Launch requires the successful preflight, fresh remote output/launch receipt, two idle GPUs and recorded master/source bytes before a single detached controller start. No package install, polling or auto-retry path was added. Network access, live resources and these remote assertions were not executed by the reviewer.

## Remaining acceptance and privacy limits

The source review is provisional same-family evidence. It does not replace real import/preflight, GPU sanity, control reproduction, frozen/Empty checks or full-run acceptance. Human review remains pending and these fitting labels are not semantic heldout evidence. Keep generated *_phrase_rows.jsonl private because it includes target labels. No raw label, reviewer evidence note or secret is included here.

## Exact source bytes actually read

| File | SHA-256 |
| --- | --- |
| `M112_CURRENT_PHRASE_TRAINING_PLAN.md` | `944fb369f6683c2b1b5aeefa899da5159f5c945a9b78a6943e195ddac49c85d7` |
| `train_m112_current_phrase_evidence.py` | `c94ebf3227cb0fdb4843f5ec05bc70a74910bf750ff17b3b73847bc1621998e5` |
| `run_m112_current_phrase_experiment.py` | `51b35c40781d7758017ddc96f16c34d7138af8b0eefe6d552f88e8cfe476afda` |
| `instance_ab_prototype.py` | `4485a7ca29a90fa13cc868c32e4e4083c351cfba7b318d341c0165c32a886e4f` |
| `train_m110_no_weak_rank.py` | `ad140851b189f363cd16a0fc5610a88720993bb89c59bfe50bc0632149ed4ef9` |
| `train_m111_explicit_category_evidence.py` | `bf73c3c661f73bac27e3861aba54409f07cd75829aaa5acec58d24ff8006e30d` |
| `train_m107_weak_semantics.py` | `a9a1f17c2b15caed6b00dfa7d3fa73abfe1f7492f73ee05bc50652e1644bccf4` |
| `train_ab_visual_control.py` | `b1f5c1a458c380eae481b5f4ebccdc619524b8eb8a275ceb5d25fc7a24f49323` |
| `prepare_m112_phrase_manifest.py` | `9d3c67271653434c6dbc33950dd1982f5d30e3f91b73de29182e3152731c38ec` |
| `analyze_train_states.py` | `2439a8d0cdb7a7d6c35bbf68aaacd28850bfb8db2211f07eb9e9e7dac66b6def` |
| `collect_ab_interface_panel.py` | `361559ef53623dc53665534243a0b4cfb078821dc852e20b34a75d14dca45086` |
| `train_fixed_visual_selector.py` | `a8c644a8ca2201c6c36a67c7146725a08ee8a9624fd9ef111da9ca596a930784` |

Private sources under `.aris/m112_current_phrase_20261003`:

| File | SHA-256 |
| --- | --- |
| `deploy_m112.py` | `2929fb367514907b3174a47798c10f952a7fba8ac1cd9530691af55c1648eadd` |
| `launch_m112.py` | `5cb7178e07a5667e6caf92111c6976cdcd9a79c41669c70b35c3fc3184ac5221` |

Companion JSON also records exact digests for the parsed private inputs and historical result/row artifacts.
