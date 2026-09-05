# Same-bundle native STTrack DepthTrack/CDTB reference

The native M39 low22 reference has a running full127 VOT evaluation, but its
DepthTrack Test and CDTB results are still missing. Historical scores on those
datasets belong to SRTrack. Complete the native STTrack reference with the
same base checkpoint, YAML, preprocessing and default template/query behavior.
This evaluation does not promote any failed association head or claim a new
trained model. M53 and subsequent learned-module advancement remain separate.

| Dataset | Sequences | Frames | Protocol |
| --- | ---: | ---: | --- |
| DepthTrack Test | 50 | 76373 | Single initialization, existing long-term PR/F-score |
| CDTB | 80 | 101956 | Single initialization, same established PR/F-score routine |

Use the native STTrack class, base checkpoint SHA256
`cacbd799115be1aaeb049cee0db89270851e3b6dd68997553b4c2c31c1104f98`,
search256/factor4, template128/factor2, fixed query window, and two templates
with native updates every50 steps when confidence is strictly above0.75.
Language is off. Keep unrounded state inside the tracker. Serialize boxes and
native confidence to six decimals, with initialization confidence1.0.

Preparation reads directory inventories, GT row counts and permitted initial
boxes. Only initial-box values enter the inference manifest. Both normalized
datasets have contiguous eight-digit one-based JPG/PNG pairs and matching GT
row counts; the inventory is checked before manifest sealing. Tracking reads
no GT files. Evaluation opens subsequent GT only after checking every output
hash and full coverage against the sealed manifest and successful exit.

The unchanged evaluator is
`/home/SRTrack_RGBD_L/lib/test/analysis/depthtrack_pr.py`, SHA256
`05879f2e732aed982fbcbebd9756ce063ed0fa945c1f6b0c04092c3e487466cc`.
Preserve its bounded VOT-region overlaps, invalid-GT semantics, global
confidence threshold grid at resolution100, per-sequence PR curves followed
by macro averaging, and maximum F-score. No confidence calibration or
post-result threshold change is introduced. Report each dataset separately.

Before either full run, use two DepthTrack Train sequences, chair01 and cube04,
through frame120 against their existing sealed native predictions. Require
the original bbox/score tolerances, actual native template updates and exact
coverage including initialization; check six-decimal output round-trip error.
This interface contract uses no new test performance evidence. Both full
datasets use the same unchanged checkpoint and runtime; no per-dataset model
choice is allowed.

Run the short contract after M53 releases GPU0. Full-run scheduling follows
available GPU capacity without stopping the active VOT reference or M53.
Seal complete predictions before CPU metric analysis. Keep reference results
separate from any later newly trained module's results; the final project
objective still requires a verified final bundle and strict VOT target scores.
