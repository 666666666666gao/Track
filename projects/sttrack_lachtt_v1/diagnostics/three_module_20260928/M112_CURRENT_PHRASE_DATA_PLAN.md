# M112: current-region evidence labels before the next semantic training

M111 completed but did not produce a development selection gain against its
own Empty condition. Its initial-only category classifier predicted supported
for all19 development cases after supervision. Do not repeat that completed run,
infer current candidate labels from it, or promote it to recursive tracking.

The next input addresses the existing Stage1 ModuleA task: explicit support,
conflict and unknown evidence for the **exact phrase actually used at the current
candidate**. The old24 candidate category strings all differ from the corrected
queries used for training. Their original labels cannot simply be inherited.

Use the same24 existing anonymous DepthTrack Train fitting events and their
existing RGB images. No new event is chosen from public evaluation or by model
performance. Preserve the prior review files. Join each of the two candidates
with its1–5 actual M107 phrase slots:202 judgments in total. Record support when
current visible RGB supports the phrase, conflict only for clearly observable
incompatible evidence, and unknown for blur, occlusion or insufficient detail.
Both candidates can support the same category. These are not physical identity
labels, nor does a low-IoU candidate automatically become a semantic negative.

The fresh-context gpt-6-astra/max reviewer receives only anonymous IDs, exact
queries and image paths. Its combined sheets contain the initial observation,
current A/B crops and unmarked Train temporal context. Original GT, candidate
indices, model choices, sequence names and old labels are excluded from its
input. The executor has already seen some historical evidence and is not blind.
Earlier/later context is disclosed as training-teacher information, never a
deployment input; it cannot make a currently hidden attribute observable.
Only RGB is shown, so the labels do not assess Depth reliability.

Review output remains model weak supervision with human confirmation pending.
The manifest compiler must match all202 rows to the exact query, slot and
private A/B-to-cached-index mapping, with every event in former-fit130. Preserve
reasons and report class counts and paired A/B support differences before
designing the next loss. There are no new current-phrase development labels.
Training agreement on these24 fitting events will not establish transfer.

Current state: preparation, fresh model review and actual manifest compilation
are complete. The reviewer viewed24 events/120 images and filled202 rows:
98supported,15conflicting,89unknown. Among101 paired phrases,29 pairs are both
supported,27 both unknown, and only10 have supported versus conflicting labels.
The remaining35 differ through an unknown label. These fitting judgments are
not independent semantic truth or held-out transfer evidence. Fresh data-source
review passed, including the explicit zero-based slot protocol. A filled local
HTML and the new CSV preserve reasons and remain human-confirmation pending.

The M112 trainer and two-card sanity/reference controller are now prepared
under `M112_CURRENT_PHRASE_TRAINING_PLAN.md`; fresh source review is pending.
No M112 GPU sanity, optimizer updates or public evaluation has been launched.
SSH43811 is presently unreachable. Restore that authorized
endpoint and complete the pending remote master-document synchronization before
GPU deployment. Any future trainer still needs fresh source review and a real
GPU sanity run under the existing experiment-bridge workflow. Keep seed2027,
the frozen visual/Empty control, fixed final checkpoint and original full
three-dataset acceptance targets. Do not scan seeds, repeat completed Full152,
or treat an old visual-capacity pass as semantic promotion.
