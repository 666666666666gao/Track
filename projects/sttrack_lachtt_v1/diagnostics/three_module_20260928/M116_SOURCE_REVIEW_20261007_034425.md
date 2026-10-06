# M116 source review

PASS — 2026-10-07T03:44:25.800922+08:00

Blocking findings: none. Non-blocking defects: none.

Requested reviewer: gpt-6-astra, max. Fresh context; same-family; acceptance provisional. Actual backend/model/effort attestation is unavailable and is not claimed. Runtime GPU checks: 0. Reviewer SSH calls: 0.

**Human text and split provenance — PASS.** Confirmed category is phrase slot 0, followed by at most four confirmed stable attributes. M113 encodes five slots plus separate object/Empty vectors. M116 checks bank/label/encoder provenance, sequence and split correspondence, and uses the unchanged M90 inference plan; no filename-derived replacement or later-frame semantic labels.

**Native frame, crop and prior reconstruction — PASS.** Current source calls the actual native get_rgbd_frame for each t0/event, samples the original six-channel RGB-D image at factor 4/output 256, then selects the first three channels. It reconstructs full-precision prior from expected_rows[frame-1], checks its float32 cast and resize factor against the M90 cache, and uses the same ceil/round crop geometry as native sample_target. Both historical M90 receipts record maximum_box_error_px=0 for every sequence. This is source/state provenance, not a historical pixel-tensor comparison.

**CLIP extraction and capacity boundary — PASS.** The hook captures the output of visual.transformer in LND order and converts to NLD. Manual CLS uses the same slice, ln_post and projection as the official/installed source within the same forward. Patch output is asserted [B,256,768]; official square 256-to-224 preprocessing introduces no outside-crop pixels. The plan identifies the extra frozen RGB capacity and treats patch projection as a grounding hypothesis.

**Spatial coordinates and padding support — PASS.** Row-major 16x16 token layout agrees with CLIP flattening. Global xywh points map through the actual rounded crop origin using align_corners=False. Candidate samples are the 4x4 bin centers; factor-2 context uses the twelve perimeter bins. Native padding validity is resized/pool-averaged to patch support and sampled with the same grid. Zero padding, coordinate-ramp and far-outside-zero assertions are present. Coverage is interpolation support, not attribute visibility or an independent segmented region.

**Frozen execution and legal observations — PASS.** Model is eval/float/requires_grad false, image forwards run under no_grad, and final parameters/buffers are checked exactly. Only original event frames plus legal initialization frame 0 are read. Collection does not read subsequent GT, instantiate an optimizer, modify a tracker, select an online candidate or write a trained checkpoint. Half feature grids, CLS, coverage and source crop origins are persisted; phrase scores use those same half features.

**Sanity-first gating, coverage and persistence — PASS.** Both GPU shards must finish sanity with exit 0, completion receipts, exact/frozen/GT guards and three events before either full shard is launched. Full collection requires 3502 total events and 152 sequences. Seed 2027, separate sanity/full directories, per-sequence features, response JSONL, process logs, exit files and completed receipts are defined. Failed runs are not retried or promoted by this source.

**CPU analysis and candidate/GT correspondence — PASS.** M114 load_inputs computes IoU from original M90 boxes without candidate reordering; its response writer preserves that vector. Actual M114 JSONL has 9117 rows, 3039 unique Empty states (2544 fit/495 development), ten IoUs per state, equal candidate IoUs across all three conditions, and matches its original receipt digest. M116 uses that same original cached box order and event key. CPU analysis excludes t0 and the 463 invalid-GT events from localization metrics, reports all eight fixed readouts, and performs no fitting or deployment. Actual original labels support 1296/223 good-poor states and 22146/3597 unequal-IoU qualified pairs in fit/development.

**Private deployment and environment — PASS.** Existing review digest gate is respected, source upload is byte-checked, prior M115 completion and both-card idleness are required, and free-space/output-existence gates precede a single detached controller. Native repository and full152-spec paths agree with the actual m98_started launch. Existing STTrack Python and local CLIP weight are used; no install, model download, deletion or fallback is introduced.

Actual local validation: Python 3.8 grammar parsing passed for 11 reviewed Python files, the private deployer and both embedded remote code blocks, using an already installed CPython interpreter. Historical JSON checks above were executed locally. No M116 collector, GPU sanity, full collection or new analysis result was executed by this reviewer.

**Acceptance limits**

- This is a source-only PASS. The reviewer made zero SSH calls and zero GPU/runtime M116 checks. GPU sanity, actual same-input CLS equality, finite cached half tensors, frozen-state equality and complete runtime coverage remain to be observed.
- The successful private inputs.json snapshot reports Python 3.8.20, torch 1.13.1+cu116 and the existing official CLIP installation. It is preparation evidence, not an M116 execution result; prior preparation errors/timeouts are not erased or recast as experiment results.
- M90 code permits a 1e-4 pixel replay difference, but the actual preserved receipts report zero maximum box error for all 152 sequences. Float32 cache checks alone would not prove discarded double bits or every historical pixel. The revised plan explicitly retains that limitation, and a future assertion failure must be inspected rather than silently relaxed.
- CLIP final projection was trained on CLS; applying it to patch tokens remains an empirical hypothesis. Self-attention mixes the full crop, and interpolated padding support is neither identity truth nor attribute visibility. Added frozen RGB capacity must remain disclosed in later matched comparisons.
- The original training events were GT-selected and human text is extra offline supervision. These are localization diagnostics on a fixed Train panel, not recursive tracking, identity/attribute verification, or formal benchmark improvements.
- Requested reviewer routing is recorded as gpt-6-astra/max. No independently verified backend/model/effort attestation is available; acceptance remains same-family and provisional.

The CLIP comparison used [official model.py](https://github.com/openai/CLIP/blob/main/clip/model.py) and [official preprocessing](https://github.com/openai/CLIP/blob/main/clip/clip.py), alongside the installed-source snapshot in private inputs.json.

The companion JSON records exact digests for all reviewed source files, the private deployer and historical artifacts. The timestamped files are written first; the fixed names are byte-identical copies.
