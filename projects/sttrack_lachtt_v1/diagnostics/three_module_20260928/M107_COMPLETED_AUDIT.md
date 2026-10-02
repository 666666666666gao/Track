# M107 completed fixed-state experiment audit

**Review:** fresh `gpt-6-astra` at `max` reasoning; same-family, provisional.
**Overall integrity verdict:** PASS within the audited cached-state scope.

The reviewer independently recomputed the stored fitting and development
JSONL summaries, compared the actual 495 M101 parent selections, checked
dataset ground-truth provenance, result and source hashes, and verified the
model-produced text-bank receipt. The Empty final produces identical selections
under Empty and weak text. The weak-text final changes 20/495 selections under
the same-weight Empty-to-text intervention, with six improved and six harmed
IoUs and a mean change of -0.001969685876798449. Both trained arms fail the
prespecified zero-healthy-break condition (two breaks each). These are
integrity-verified negative scientific results, not an experiment-integrity
failure.

| Check | Verdict | Evidence and boundary |
| --- | --- | --- |
| A. Ground-truth provenance | PASS | `prepare_train_states.py` reads hash-bound dataset `groundtruth.txt`; `train_ab_visual_control.py` calculates candidate IoU against it. Model-produced A/B labels supervise fitting only. |
| B. Score normalization | PASS | IoU uses intersection/union; success is IoU≥0.5 and mean IoU uses event count. Oracle candidate coverage is separately identified. |
| C. Results and metrics | PASS | Both arm JSONL files, content interventions, source/prototype/final hashes and private weak-bank receipt match their summaries. |
| D. Dead code | PASS | Evaluation calls are executed; semantic gradients are nonzero for the text sanity arm. |
| E. Scope | PASS, narrowly | 130 fitting sequences/2,544 valid sampled states, 22 Train development sequences/495 sampled states, seed2027. No recursive or external-benchmark claim is supported. |
| F. Evaluation type | `real_gt` | Dataset boxes assess cached-state localization; model annotations are weak training labels, not evaluation truth. |

The reviewer did not run checkpoint replay against raw cached features. The
original per-child stdout logs are not in the local public result directory;
the controller completion log is preserved privately and binds the 80.2894 s
completed run. The private text bank and model-produced review CSVs are not
published. This audit does not establish that the reviewed descriptions are
semantically correct, nor any DepthTrack Test/CDTB/VOT performance.
