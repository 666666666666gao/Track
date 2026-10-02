# M108 prelaunch source and integrity review

**Verdict: PASS for the reviewed prelaunch source.** No launch-blocking code defect or unsupported claim of completed M108 results was found. This is a fresh Codex review with `review_independence: same-family` and `acceptance_status: provisional`; it is not a runtime or scientific-success verdict.

The reviewer read the supplied sources and historical results directly, inspected the added controller and checkpoint round-trip check, and ran local standard-library AST, hash, and JSON arithmetic checks. No Torch import, model execution, GPU job, server command, or experiment-source edit was performed. Only this report and its JSON companion were written. M108 has not run within this review.

## 1. Freeze, gradients, and Empty parent preservation — PASS

- `train_m108_frozen_semantics.py:12–13,68–79` strictly loads the specified M101 parent, freezes all parameters, enables only the text projection/normalization, phrase slots, two text attention modules, phrase feature MLP, and semantic/phrase readouts, and zeros the two bias-free readout matrices. The optimizer receives only parameters with `requires_grad=True`. Selection, quality, observation, geometry, visual attention, candidate attention, and visual encoding remain frozen.
- The training forward/loss/backward at `train_m108_frozen_semantics.py:91–110` is outside `no_grad`; the frozen selection MLP remains on the differentiable path from the loss to the semantic increment (`instance_ab_prototype.py:112–115`). Freezing its weights does not detach its input. Zero readouts initially block internal text gradients, but subsequent updates can open that path; the third sanity update explicitly requires nonzero text-projection and semantic-readout gradient norms (`train_m108_frozen_semantics.py:106–109,130–132`). All existing gradients must be finite, and every frozen parameter must have no gradient (`:103–105`).
- Empty uses the same per-sequence mask for both branches (`train_m107_weak_semantics.py:11–19`; `instance_ab_prototype.py:84–91`). Both attention modules have no configured dropout (`instance_ab_prototype.py:33–34`). Therefore identical Empty inputs produce zero centered semantic and phrase deltas; the bias-free readouts preserve the visual selection and quality functions (`instance_ab_prototype.py:100–101,112–116`).
- Before optimization, the code snapshots Empty scores/quality for all 2,544 fit and 495 development states, and compares every development selection to the saved M101 selection list (`train_m108_frozen_semantics.py:27–35,76–78`). After optimization it compares every frozen parameter, every buffer, and every Empty score/quality tensor exactly (`:113–118`). The runtime score/quality comparison is initial-versus-final within the loaded arm; the historical saved-parent comparison is the 495 selections. Exact parent-function preservation is additionally supported by the unchanged visual dependency path, rather than a nonexistent historical score-tensor file.

## 2. Generic active control and unchanged weak inputs — PASS

- `prepare_m108_generic.py:13–17` pins the original M107 bank and CLIP encoder hashes. Lines 19–27 encode the nonempty generic prompt with the frozen encoder and add only the generic vector and provenance metadata to the loaded bank. The original text tensors, Empty vector, masks, sequence ordering, split entries, and label binding are not rewritten.
- `train_m108_frozen_semantics.py:16–24` broadcasts that vector into the original valid slots and retains the exact mask. The weak-text arm passes the original bank to the unchanged M107 input helper. Its helper and prototype SHA-256 values exactly match those recorded by the actual M107 result (`m107_completed/train_weak_text/result.json:5–6`). Both arms use the same seed, parent, optimizer, batches, losses, and fixed epoch count (`train_m108_frozen_semantics.py:60–90,94–102`).
- A local aggregate-only inspection verified the input label file's SHA-256 against M107's recorded binding (`m107_completed/train_weak_text/result.json:14`): 152 distinct sequences, 130 fit and 22 development, no sequence overlap, 11 unique accepted fit-only pair events, and 13 ignored uncertain events. Candidate indices are distinct and within the ten-candidate range. The development sequence set exactly matches M101's saved rows. No private descriptions or per-event choices are reproduced here.
- The generic arm has a nonempty semantic input and the same trainable modules; it is not the Empty cancellation control. Actual nonzero internal gradients remain a required, not-yet-observed sanity result. Because masks are deliberately preserved, this comparison tests token content conditional on the existing slot masks; it does not remove all information carried by mask length.

## 3. GT, training split, and exact denominators — PASS

- Dataset GT comes from hash-checked `groundtruth.txt` coordinates, including finite/positive-size validity checks (`prepare_train_states.py:69–78,102–105`). It is not generated from the model's candidate predictions. The source split is by sequence, with the original first 130 fit sequences and remaining 22 development sequences checked against the earlier specification (`:39–45,62–64`).
- Feature collection reads the separate inference plan, images, and native tracker outputs; the GT-dependent event-label file is not loaded (`collect_train_states.py:56–65,83–84,93–147`). The current loader verifies cached feature hashes, split agreement, and absence of loaded GT/text in the feature receipts (`train_ab_visual_control.py:25–28,47–59`). Current-frame GT is joined afterward only to compute candidate IoUs (`:60–75`). `batch` explicitly excludes the IoU tensor from model input (`:86–94`); geometry uses prediction scores, candidate geometry, and the prior predicted box (`collect_ab_interface_panel.py:52–62`).
- Model-derived weak pair choices enter only the fit loss, with fit membership assertions and exactly 11 distinct accepted events (`train_m108_frozen_semantics.py:80–87,94–102`). Their uncertain or unconfirmed status is not converted into evaluation GT (`:126`; `prepare_m107_weak_text.py:12–14,30`). Development annotations are not used by the optimizer.
- `load_inputs` skips invalid current GT and requires exactly 2,544 fit / 495 development states (`train_ab_visual_control.py:60–61,81–82`). `summarize` reports raw counts at IoU >= 0.5, with `valid_gt=len(values)`, and mean IoU divided by that same group's state count (`:114–128`). No reported metric is divided by a model-output maximum, mean, or oracle score. Native-gap preservation normalizes a training loss by its number of eligible candidate pairs, not an evaluation metric (`:148–153`).

| Development group | Exact valid-state denominator |
| --- | ---: |
| All | 495 |
| Healthy | 264 |
| Transition | 127 |
| Intermediate | 69 |
| Late low | 41 |

The strata overlap, so the subgroup counts must not be summed to create a new denominator. The reviewer recomputed all M101 development summaries from all 495 unique saved rows and obtained exact equality, including mean IoU. These rows cover exactly 22 sequences (`m101_completed/train_weight1/result.json:170–222` and the subsequent row array).

## 4. Invocations, completion, and saved outputs — PASS

- There is a concrete executable chain: controller -> pair driver -> generic preparation -> both sanity children -> both training children. `M108_CONTROLLER.py:6–12` makes one child invocation, persists its actual terminal exit, and exits with the same code. It contains no retry or restart loop. Its durability still depends on how the controller process is launched; this review did not launch it.
- The driver creates a new output directory (`run_m108_frozen_pair.py:34`), checks preparation exit (`:35–39`), waits for both sanity children, and requires each sanity receipt to report three updates and no saved checkpoint before launching training (`:40–44`). Each phase invokes the actual trainer entry point with explicit arguments, separate GPU assignments and separate logs; both child exit codes must be zero (`:7–28`). Each final training receipt must have 480 updates (`:45–47`), matching 12 * ceil(2544/64).
- The trainer writes fit rows and three complete development content-condition row files from each same final model (`train_m108_frozen_semantics.py:134–140`). `evaluate` is actually called and passes its generated rows to the shared metric function (`:38–50`). The final is fixed by epoch count, with no development-based checkpoint search (`:88–111,147`).
- The final checkpoint is saved, reloaded onto CPU, and every state tensor is compared exactly before recording its successful round trip and SHA (`train_m108_frozen_semantics.py:147–151`). The result JSON is written afterward (`:153`), and the pair receipt is written only after both successful final children (`run_m108_frozen_pair.py:45–55`). The added save/reload logic is included in the reviewed source hash: `f66e862477da660e6d68e5f57f066d8acd83adbc2d85923421a41e11b31224e3`.
- These are verified invocation and persistence paths in source, not evidence that the new experiment has already run. Missing future M108 runtime outputs are expected at this stage and are not classified as fabricated results.

## 5. Historical evidence and scientific scope — PASS with required interpretation limits

The historical motivation is supported by actual files, not merely by the plan. The reviewer recomputed the fit and both content-condition summaries for both historical M107 finals from their saved JSONL rows. Every summary matched its result JSON exactly. For the weak-trained final, mean development IoU is 0.5314440731279025 under weak text versus 0.533413759004701 under Empty, a difference of -0.001969685876798466; both conditions have two healthy breaks (`m107_completed/train_weak_text/result.json:222–346`). The separately Empty-trained final also has two healthy breaks. Both historical runs contain 12 epochs, 480 updates, and 11 accepted-pair calls per epoch in the weak-input trainer; the M101 parent-hash linkage also matches.

The following limits must remain attached to any later interpretation:

1. Evaluation type is **real_gt fixed-state development**, with **model-derived weak supervision** during fitting. Event sampling is GT-stratified (`prepare_train_states.py:79–90`), and this is the reused Train development panel. It is not an untouched public-test estimate, full recursive tracking result, identity-label accuracy, or human evaluation.
2. The four saved gates compare against the native candidate, not against M101 or the generic arm (`train_m108_frozen_semantics.py:143–146`). Passing them alone cannot establish semantic benefit. Compare the saved Empty, generic and weak-text results, with the immutable parent as reference, before making that claim. The code saves sufficient per-state parent and selected values for that comparison (`:45–49,137–140`); it performs no automatic promotion.
3. One seed and two fixed training arms support a bounded development comparison. The unchanged plan already avoids claiming a new architecture, calibrated phrase-channel evidence, or public/recursive success (`M108_FROZEN_SEMANTIC_PLAN.md:6–15,38–43`). No unsupported stronger scope claim was found in the reviewed M108 artifacts.
4. Actual CUDA gradient checks, frozen-state checks, final round-trip checks, remote input hashes, and final metrics remain pending execution. The local deterministic checks validate source and historical arithmetic, not these future runtime outcomes.

## Integrity checklist

| Check | Status | Evidence |
| --- | --- | --- |
| A. GT provenance | PASS, source scope | Dataset coordinates are separate from model inputs and weak supervision; see section 3. |
| B. Score normalization | PASS | Raw threshold counts and true valid-state denominators; no self-normalized performance. |
| C. Result existence / consistency | PASS for historical evidence; M108 pending | M101 and M107 row arithmetic and hash bindings checked; no completed M108 result asserted. |
| D. Dead code / invocation | PASS | Real subprocess and function call chain, with required output writes; see section 4. |
| E. Scope | PASS with stated limits | Train-only, fixed-state, single-seed development comparison; gates do not establish text benefit alone. |
| F. Evaluation type | real_gt | Model-derived weak labels are training supervision, not evaluation truth. |

**Required source corrections: none found.** Retain the existing sanity-before-training sequence and report its actual outcome. The JSON companion records the audited input hashes and deterministic checks.
