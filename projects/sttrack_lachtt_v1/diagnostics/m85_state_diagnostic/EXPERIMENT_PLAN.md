# M85 fixed-state diagnosis of M84

Scope: read-only diagnostic, no training, no deployment policy change, seed2027. Fixed three posthoc cases: mobilephone02_indoor (native-success damage), car02_indoor (prolonged failure), ghostmask_indoor (positive comparison). 2616 noninitial tracking calls in total. These selected cases are not an unbiased estimate of failure prevalence and cannot promote M84.

Use the M84 final Category checkpoint and its frozen code, initialization boxes, Category/Empty/Swapped banks and default policy. Replay each selected full sequence in original order, assert every actual Category bbox and score exactly matches its sealed full evaluation. Do not change chronological state or use GT resets. No subsequent GT is loaded during replay.

Capture current RGB, depth, fused features and initialization reference at the adapter boundary. Category is the only state-committing trajectory. Empty, Swapped and native Head receive the same captured current state but do not commit state. Assert Empty/native score, size and offset maps exactly equal. Assert alternative reads do not change bbox, query tensors or template identities. This does not reconstruct independent native or Swapped trajectories after a divergent history.

Save per-frame actual previous bbox, integer crop origin and side, image dimensions, actual default template-write eligibility, selected raw/Hann boxes, maxima, indices and normalized native spatial KL. Also save finite float32 dense score/size/offset arrays for Category/Swapped/native and the actual Hann window. Empty is identical to native, so do not duplicate its dense arrays. Shapes: [frames-1,5,16,16], channels score,width,height,offset_x,offset_y. Source and all output artifacts are hash-bound.

Preflight: first 101 tracking positions of the first case, in a separate output directory. Only after this exact interface replay succeeds run the complete three sequences. Expected complete runtime roughly 3-6 minutes plus startup, based on prior 22-sequence replay; this is an estimate. Reserve at least 200 MB scratch space; no new large weights, keep both Qwen models.

Posthoc analysis is separate, after the complete prediction receipt seals. Use dataset-provided GT with valid-coordinate/positive-size filtering; exclude initialization and break low-overlap runs at invalid GT. Report local crop center/full-box coverage separately from any dense-decoded IoU>=0.5 candidate capacity. Compare raw/Hann selection, all candidate capacity, and same-state content changes on all selected frames, including both rescue and damage. Candidate existence is hindsight capacity, not achieved trajectory recovery. Confirm selected decoded boxes against saved online boxes before using the dense decoder.

Hypotheses to separate, without asserting a cause before observation:
1. Immediate content-dependent selection: same-state alternate text/native output can select a correct candidate where Category is wrong.
2. Inherited observation/state failure: target is outside the crop, or no dense decoded candidate is correct; same-state replacement cannot supply the missing evidence.
3. Final spatial competition: raw or another dense candidate is correct while Hann selection is wrong; also count healthy frames that raw selection damages.

The replay localizes these conditions. It cannot isolate the causal contribution of earlier template writes or query updates without a later separately defined state intervention. Do not equate invalid GT with absence or rotation. Do not modify Hann, thresholds, captions or memory in this experiment. Preserve M84's failed performance verdict.
