# RGB-D initialization-description review

This static GitHub Pages site contains one review case for each DepthTrack Train
initialization (152), CDTB sequence initialization (80), and VOT-RGBD2022 legal
initialization anchor (1,765). The red rectangle is the protocol's initialization
box. Neighbor frames and sampled whole-sequence videos have no future boxes.
The 25-fps video timeline is a review playback setting, not a measurement of
the original capture rate. Clear videos retain the full sequence duration and
sample eight images per review second at 640-pixel width (H.264, CRF 28).

The default board remains the lightweight 960×565 JPEG. Click the clear-image
button to load a 1920×1130 JPEG, regenerated from original dataset frames at
quality 88. Inspect the initialization target, full boxed frame, neighbor
frames, or the whole board at 1x to 3x display scale. Only one case is rendered
at a time; clearer images and videos load only on request. Reviewer IDs, case
IDs, browser answer keys and CSV fields remain unchanged.
Display enlargement does not create detail absent from the source image.

The displayed category and attributes are provisional automatic descriptions.
Reviewers choose supported, conflicting, or uncertain, can correct the category
and attributes, and export their own UTF-8 CSV. Answers are saved only in that
reviewer's browser; the site does not silently aggregate or overwrite answers.
Use a distinct reviewer ID and return the exported CSV to the project owner.
Filters and numeric ranges allow people to divide the 1,765 VOT anchors.

Only DepthTrack Train review may inform future training. CDTB and VOT review is
for input auditing and error analysis; its labels and later frames must not
enter training or model selection. Inference descriptions use only each legal
initialization observation.

`build_assets.py` reads the frozen bfloat16 CDTB/VOT generation plans and
records, verifies the record-file hash and each image binding, then makes
boards, manifests, and review videos. `build_depthtrack_assets.py` uses the
existing DepthTrack Train initialization manifest and creates the same format.
No future tracking annotations are included in site assets.

`build_clear_assets.py` regenerates the clearer evidence for all three datasets.
It retains existing preview paths and descriptions, verifies original
initialization-image hashes and source-manifest bindings, and writes versioned
detail/video files plus build receipts. Later frames remain unmarked review
evidence and do not enter external-dataset training.
