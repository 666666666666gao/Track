# M115 source review

Verdict: **PASS**. No unresolved blocking or nonblocking finding.

Reviewed at 2026-10-07T02:40:58.334665+08:00. Fresh context (`fork_turns=none`); requested `gpt-6-astra` / `max` spawn confirmed by the parent; serving backend not independently attested. Independence is **same-family**, acceptance **provisional**.

This is a source-only predeployment review with local stdlib checks of completed evidence. The reviewer did not use SSH, access credentials, import Torch, run a neural network, start an experiment, or mutate experiment code. M115 runtime acceptance and scientific benefit remain pending.

One source issue was corrected during review. The original optimization loop called `model.train()` whereas its frozen Parent reference snapshots used eval mode. The recorded environment is Torch1.13.1+cu116, whose candidate self-attention can change execution path between these modes. The executor changed the loop to `model.eval()` at `train_m115_semantic_choice.py:72`, with gradients enabled and the exact invariant retained. The final source and plan disclosure were reread. This is source-level correction, not a claim of a reproduced GPU failure. [Official PyTorch1.13.1 implementation](https://raw.githubusercontent.com/pytorch/pytorch/v1.13.1/torch/nn/modules/activation.py).

- The independent head is131->64->1, with no last-layer bias and zero last-layer weights. It trains8,512 head parameters plus91,395 text-interaction parameters (99,907 total). Frozen visual/quality outputs and consistent selected box/score/quality/feature fields are preserved structurally and checked in source.
- Full-minus-zero-semantic scoring is centered over the10 candidates. Relative CE uses normalized GT-IoU mass only among IoU>=0.5 candidates; rows without a good candidate contribute no invented positive. Preservation uses the actual reliable Parent argmax against IoU<=0.1 candidates. These are localization labels, not physical identities.
- All three fresh arms share seed2027, initialization, bank, split, batch64, AdamW3e-4 and12 epochs/480 updates. Each receives a separate3-update sanity first. The human pair precedes the generic arm; failures stop progression without retry or promotion.
- GT comes from dataset `groundtruth.txt` through the existing cached-box IoU computation. The model input adapter removes IoU. Actual human input has152 unique Train sequences,704 phrases and130fit/22development; no Test/CDTB/VOT optimization or new current-candidate semantic labeling is introduced.
- Source includes the3039-state Empty comparison, parameter/buffer freeze, finite/nonzero sanity-gradient, nonempty quality, selected-field and serialization checks. The final checkpoint is fixed; content interventions are evaluations, not checkpoint selection. New selection logits are not installed as template-update probabilities.
- Deployment, one launch+240s observation, terminal-only copying and three-arm stdlib recount were reviewed. Binary final weights stay private. These private helpers were not executed.

Direct evidence checks read18,234 stored M114 responses and replayed each arm's4,029 original M113 choices/GT IoUs. All11 stored exit files checked are0. M113 human and own Empty both hit272/495; M114 human development response has92.1824178887% common energy, with19 improving and204 shrinking good/poor margins. No-good counts are730fit/190development in the10-candidate set. The recorded M47 cross-frame auxiliary multi-positive comparison retained default-priority action labels and failed its completed recursive gate. This evidence supports the proposed diagnostic comparison, not a unique causal BCE explanation or novelty from a loss alone.

The actual Parent preservation set contains2,271 fit pairs versus2,165 native-candidate pairs. The plan correctly discloses this change, so M113->M115 is not a single-factor loss ablation. The two human M115 arms provide the matched localization-objective comparison.

Local Python3.8 AST parsing passed for16 source/helper files and3 embedded remote blocks. Extracted pure stdlib reporting functions reproduced8 completed summaries and4 own-Empty paired summaries. The JSON records57 directly reviewed source/evidence digests and all3 private-helper digests; it excludes this review, MANIFEST and the mutable project master.

Actual M115 GPU sanity, full480-update results, content benefit versus own Empty/generic, and healthy-state preservation remain to be verified. Fixed-state success would still require subsequent recursion and actual same-final official evaluation; no formal metric or three-module contribution is established here.
