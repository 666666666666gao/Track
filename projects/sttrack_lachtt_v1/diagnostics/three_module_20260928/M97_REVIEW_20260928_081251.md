# M97 predeployment code and scientific-integrity review

- verdict: PASS after two collection corrections
- reviewer: fresh Codex reviewer, gpt-6-astra / max
- review_independence: same-family
- acceptance_status: provisional
- scope: source review and CPU synthetic checks only; no deployment, GPU work, actual-input panel run, public evaluation, or private candidate-answer access

No remaining concrete blocking defect was found in the updated implementation. This clears the source-review gate for the planned one-case collection smoke. It does **not** establish actual-input or native-equivalence success, and does not clear full training or recursive promotion.

Reviewed `M97_AB_INTERFACE_PLAN.md`, `instance_ab_prototype.py`, `collect_ab_interface_panel.py`, `probe_ab_interface.py`, the M90 preparer/collector and IoU helper, the M95 initial-origin collector, the staged experiment plan, and the local overlay's native tracker/model and candidate/spatial observation APIs. The existing native `sample_target` implementation was read from the saved M84 source snapshot. The executor separately confirmed its padding convention in the deployment source.

## Collection issues corrected during this review

1. **Do not re-decode exact proposals using a float32 cached resize.** M90 computes `256 / ceil(sqrt(prior_w * prior_h) * 4)` as a Python float for candidate decoding, but stores `resize_factor` as float32. The initial M97 implementation converted that rounded tensor back to a Python float before demanding exact box equality. That changes the decoding arithmetic and can invalidate the exact comparison. Updated collector line 161 calls native `sample_target(image, prior, 4., output_sz=256)` to recover the actual resize from the replayed prior. Initial references already use that function. The exact M90 box/RoI assertions remain intact; their tolerance was not weakened.

2. **Use actual crop padding support.** The initial coordinate-only mask checked nominal image bounds. Native `sample_target` uses `max(x2 - W + 1, 0)` and its vertical analogue when padding, so nominal image bounds do not describe all pixels retained by this implementation. Updated collector lines 24-38 sample the actual resized attention-padding mask with nearest `grid_sample`, zero padding, and `align_corners=False`, at the same continuous RoI locations. The initial and event crops both supply their own native masks. The revised plan correctly calls this support at each sample location, not validity of an entire contextualized ViT token or its receptive field.

These changes fix the existing collection contracts without adding a fallback or general compatibility layer.

## Correctness findings

- **The collection is isolated and labels remain separate.** Filtering the fixed M90 inference manifest to fit rows before taking the first eight, then slicing `[shard::2]`, implements the planned outcome-independent sequence choice and two four-case shards. The earliest stored event is replayed from legal initialization. The collector opens no subsequent GT, phrase annotations, text bank, or candidate answers. Native prediction rows are used only for trajectory equivalence. The separate probe obtains localization targets from the existing Train GT labels and ignores invalid current GT.

- **Initial and current observations follow the existing APIs.** The t0 auxiliary call uses the legal initial RGB-D image and box, passes no historical query, and does not assign its returned query. Tracker frame, box, and query state are checked immediately afterward. Every real prefix frame is checked against the native trace. The t0 search RoI must exactly match M95; the event's ten half-precision RoIs and float32 proposal boxes must exactly match M90. Existing 4 by 4 center sampling is retained. Selecting the twelve perimeter cells of a 2x expanded box produces the specified outer ring. Raw nonzero depth fraction is recorded within each proposal and is correctly described as a support descriptor, not calibrated reliability or an identity label.

- **The requested prototype paths are present.** RGB and depth have separate projections; local sample positions and initial/current/region/context roles remain explicit. The visual branch reads initial/context references. Phrase queries read concatenated initial/current/context tokens, and current candidate tokens read the bound phrases. Both full and encoded-empty content use the same modules and masks. Their difference is exactly zero for identical empty content. Candidate attention contextualizes the visual candidate set; semantic and phrase residuals are added afterward for selection. This is a scoped wiring prototype, not evidence that the attention or three-state outputs have learned instance semantics.

- **Selection and quality responsibilities are separated.** The selection residual's final layer starts at zero, retaining native log-response scores and native candidate zero. Geometry is never changed. Localization quality uses only the visual branch. Box, selection score, quality, and feature are gathered with the same final index. The observation head is absent from the loss; its parameters receive no gradient and remain unchanged in the CPU check.

- **The probe is a two-step gradient check against GT.** Selection and quality BCE targets are actual candidate-versus-Train-GT IoUs, not native predictions or physical-identity answers. The first update opens the zero selection head; the second checks finite nonzero gradients through both interaction directions and the other named modules. The phrase-evidence head can receive indirect selection gradients, as the revised plan now states; it has no semantic-state supervision or calibration. No checkpoint is saved, and the probe does not execute a tracker action or official evaluation.

## Verification performed by this reviewer

The updated three Python files pass AST parsing. On existing **PyTorch 1.13.1+cpu**, an eight-case, ten-candidate **synthetic** input check exercised the prototype and two throwaway optimizer steps. Native zero residual, exact empty-versus-visual selection, zero empty semantic/phrase increments, quality independence from text, selected-field alignment, finite outputs, and the six requested nonzero gradient paths passed. Candidate permutation maximum errors were 0 for selection and 5.960464477539063e-08 for quality. Observation-head parameters were unchanged.

A separate **synthetic CPU border-support check** used the existing `sample_target` and revised mask function: an interior box had 16 valid samples; samples in the native right/bottom padding had 0; an outside-crop box had 0; the perimeter output had the required 12 entries. This specifically checked the revised padding-mask path.

These checks used artificial tensors/images. They did not load the real M90/M95 tensor caches or category bank, execute the deployed GPU kernels, or verify the actual eight cases. They cannot replace the following gate.

## Required remaining runtime gate

1. Run the planned one-case native collection smoke before the two full shards. It must finish with finite tensors, expected local/ring shapes, unchanged t0 public state, exact M95 t0 RoI and M90 event box/RoI checks, and every prefix bbox/score check passing. Current prefix tolerances are 1e-4 pixels and 1e-6 score. A failure must be diagnosed from its assertion/trace before retrying; do not waive exact-reference assertions to obtain a pass.

2. Only after that smoke passes, collect both four-case shards. Inspect the receipts for eight distinct prescribed fit sequences, each at frame 10, totaling 80 native prefix calls, with every native/reference assertion passing. The smoke's duplicated case is excluded from that eight-case panel. Expected shapes per four-case shard include `initial_rois=[4,2,16,768]`, `initial_context=[4,2,12,768]`, `candidate_rois=[4,10,2,16,768]`, `contexts=[4,10,2,12,768]`, corresponding 16/12-location boolean masks, `geometry=[4,10,13]`, `base_scores=[4,10]`, and `boxes=[4,10,4]`.

3. Run `probe_ab_interface.py` on those two actual shards and the prescribed unverified category-only bank. Require successful artifact loading, valid fit GT targets, exact active-attribute emptiness with the original five-slot masks, all initial/post-update/permutation/alignment/finite assertions, and finite nonzero second-step gradients for all six recorded modules. The resulting `interface_sanity.json` must report `complete_interface_sanity_only` and exactly two localization-IoU optimizer steps. Save the collection receipts with it.

No extra non-blocking code changes are requested. Passing this gate establishes only that the isolated A+B interface works on this small actual-input panel. Independent reviewed phrases and candidate-instance labels, the content-versus-visual development gate, and any later recursive or official evaluation remain separate requirements. This review is same-family and provisional, not cross-family acceptance.
