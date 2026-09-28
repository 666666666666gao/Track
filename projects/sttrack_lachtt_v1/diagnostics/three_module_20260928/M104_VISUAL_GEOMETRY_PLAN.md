# M104: isolated B local visual geometry refinement

M103's actual complete CPU diagnosis is the motivation. On fit130 alone,
292events have the target center inside yet no full256 IoU>=0.5 box.
GT-size-only/full256 reads218hits, GT-center-only78. Native Top-10 equivalents
are100/46. Of these292events,205meet the predeclared entire-GT-in-crop AND
Top-10 maxIoU>=0.1 eligibility. This supports checking local visual geometry;
it does not promise learned recovery. Keep the minimum10pixel rule unchanged.

Freeze the complete M101 weight1 final, including selection, quality and all
semantic/observation parameters. Use Empty input and retain its selected index.
Add one separate visual refiner: preserve the ordered32RGB-D candidate and
24context tokens from the existing frozen evidence encoder; mask invalid
tokens; project64->16 and flatten56tokens plus13native geometry inputs into
a64hidden MLP and four corrections. Zero-initialize the last layer.
No phrase labels, online Qwen, C, search expansion or tracker action.

Decode center offsets relative to native width/height and logarithmic size
ratios, then apply the same image/minimum10clipping. Zero corrections must
reproduce all3039events' native candidate boxes before optimization. If this
actual arithmetic fails, retain the failure and fix only the demonstrated issue.
Selection logits/indices remain exactly the frozen parent's throughout.

Fit exactly one highest-native-IoU candidate for each eligible fit event
(2033expected); do not turn all incorrect candidates into copies of the target.
Eligibility and target assignment use Train GT only in the fitting loss.
Forward/evaluation refines all ten candidates in all3039events; no inference
GT gate. Use SmoothL1 on normalized center/log-size targets plus2GIoU with
the existing native box utility. AdamW3e-4, batch64, seed2027, fixed12epochs,
384updates; final checkpoint only, no development-best or threshold search.
Only59540new parameters optimize; do not retrain the parent.

A GPU sanity performs two updates on64fit events, checks exact zero-output,
nonzero final-layer and second-step input-projection gradients, finite values,
unchanged parent tensors/scores, no development evaluation/checkpoint.
After passing, GPU0 fits the fixed budget; GPU1 independently reads the frozen
parent on all3039events. This reference run is useful because the old parent
report retained495development rows but not all2544fit choices.
Both arms use the same cached inputs. No duplicated multi-seed training.

Report before/refined selected IoU, native/refined Top-10 oracle, selected index,
all events and overlapping strata; gains/harms against the parent, not just
against native. Acceptance checks: refined selected correct count and meanIoU
exceed parent, healthy newly broken events zero, transition correct count at
least parent. Keep failed checks and all raw rows. Final reload must reproduce
every saved row. Passing only establishes Train fixed-state visual capacity,
not semantic benefit, recursion, Full152 or public-test acceptance.

Estimated actual fitting/readout under2minutes based on M101's48.8second pair;
cache loading and initial complete encoding are included in measured runtime.
Source review precedes deployment; no unchanged retry. Existing nine-target
goal, independent semantic evidence requirement and all earlier results remain.
