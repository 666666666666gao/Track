# M108 frozen visual parent and provisional text training

Completed on 2026-10-02 on SSH43811. GPU0 trained the generic nonempty `object`
condition; GPU1 trained the model-reviewed initialization descriptions. Each arm
used seed2027, the same M101 final parent, 130 fit sequences / 2,544 cached
states, 12 epochs, batch64 and 480 optimizer updates. The 22 Train development
sequences contain 495 cached states. This is not full recursive training or an
official DepthTrack Test/CDTB/VOT evaluation.

Only text projection/interaction and centered semantic/phrase readout parameters
were optimized. Visual projections, candidate interaction, visual selection,
geometry, quality and observation parameters stayed frozen. The frozen
selection layers retain gradient propagation to the semantic inputs. Both arms
use the same 11 model-prefill A/B pairs; 13 uncertain pairs are ignored. These
are provisional model labels, not independently confirmed human identity truth.
The generic arm preserves phrase masks and exercises the nonempty path.

The one-shot controller ran preparation, concurrent three-step sanity checks,
then concurrent training. All seven terminal exits are zero; the actual driver
elapsed time is 60.616016 seconds. Final checkpoints and all condition rows were
saved. The actual remote artifact hashes, source hashes, final checkpoint hashes
and independent JSONL recount pass. GPU assertions verify all frozen parameters
and buffers unchanged, exact save/load state equality, and identical Empty
scores and quality across all 3,039 states before and after training.

## Stored development results

Correct selection means selected candidate IoU >= 0.5. Rescues/breaks compare
with the same-state native candidate. Healthy and transition are pre-existing,
overlapping strata; their counts must not be summed as exclusive partitions.

| Fixed final / input | Correct / 495 | Mean IoU | Rescues / breaks | Healthy correct / 264 | Transition correct / 127 |
|---|---:|---:|---:|---:|---:|
| Same-state native | 268 | 0.524770723 | - | 264 | 4 |
| Either final / Empty | 272 | 0.530707698 | 5 / 1 | 264 | 4 |
| Generic final / generic | 274 | 0.532438554 | 9 / 3 | 262 | 5 |
| Generic final / reviewed text | 272 | 0.530010909 | 5 / 1 | 264 | 4 |
| Reviewed-text final / generic | 272 | 0.531629981 | 5 / 1 | 264 | 4 |
| Reviewed-text final / reviewed text | 272 | 0.530258504 | 5 / 1 | 264 | 3 |

The generic arm fails the preset zero healthy-break gate; the reviewed-text arm
fails the transition-correct-at-least-native gate. Neither advances to recursive
integration or public benchmark testing. No preset gate, checkpoint budget or
threshold was changed after seeing the results.

Within the reviewed-text final, reviewed vs generic changes 13 selections,
improves IoU in 6 states and harms 3, but changes mean IoU by -0.001371477.
Reviewed vs Empty changes 15 selections, improves 7 and harms 3, with mean IoU
change -0.000449194. Small positive changes do not outweigh the larger harms.
The generic final likewise has no positive reviewed-text mean-IoU increment.

This experiment isolates semantic edits from shared visual parameter drift. It
supports exact cached-state Empty preservation; it does not support a net
semantic benefit, reliable phrase support/conflict/visibility predictions, or
official recursive performance. The phrase channels have no independent
support/conflict/visibility labels. Candidate IoU is localization supervision,
not physical-identity ground truth.

## Follow-up

Preserve this negative result rather than repeating the same budget or promoting
a failed gate. Use only fit-state evidence to inspect alignment between the 11
weak A/B judgments, localization targets and phrase observability before changing
the semantic objective. Keep human review pending and version the later confirmed
labels as a separate experiment. CDTB/VOT human review and future frames remain
audit-only and cannot enter training or model selection.

Deterministic recount: `M108_CPU_RECOUNT.json`. Full integrity review:
`M108_COMPLETED_AUDIT.md/json` (fresh same-family reviewer; provisional assurance).
Raw artifacts: `m108_completed/`. M108 produced no new official benchmark
evaluation; no global artifact-immutability audit is claimed.
