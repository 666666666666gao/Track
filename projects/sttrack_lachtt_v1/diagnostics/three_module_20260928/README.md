# A/B/C development on DepthTrack Train

This folder contains isolated development diagnostics. No new official tracker
or new nine-metric result is represented here. M67/M82/M89 Full152 and their
three-dataset official evaluations are complete; none meets the joint target.

- M90: native-equivalent Train state and candidate cache,152 sequences/3502events.
- M91: fixed-state visual quality control; no recursive promotion.
- M92: private human-review sheets. Original seeded A/B assignment was
  reconstructible; private v2 reissued with shuffled IDs/order/sides. Answers
  and private mappings are not published.
- M93: nominal target/unrelated-mask Qwen evidence; actual painted areas differ,
  so it does not establish a matched-area causal effect or correct semantics.
- M94: all96Qwen choices wereA; teacher gate failed.
- M95: legal t0 reference origins; direct cosine remains below native selection.
- M96: learned t0-search control selects271/495 versus270 for the original
  visual control, but lower meanIoU/transition result prevents promotion.
- M97: isolated token-level bidirectional A and candidate-set B pass actual
  eight-case wiring/gradient checks. See [completed evidence](m97_completed/RESULT.md)
  and [plan](M97_AB_INTERFACE_PLAN.md). No semantic performance claim, full
  training, recursive action, C deployment, or public evaluation.

Reliable Train phrases, candidate physical identities and semantic visibility
still require reviewed labels. IoU quality targets do not provide those labels.
The next semantic content-versus-visual experiment must use defensible labels
and preserve the staged development gates.
