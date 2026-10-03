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
GPT-6 Astra Pro initial reviews have been imported for all 1,997 cases, including
the user's DepthTrack 152-row CSV and CDTB/VOT 1,845-row CSV. Each case keeps its
original generated description alongside the model's proposed category,
initially supported attributes, uncertainty and separate later-frame notes.
The source files are hashed in `data/model_review_receipt.json`;
`data/gpt_initial_reviews.csv` is the normalized downloadable initial-review table.
GPT opinions are not promoted to human confirmation. Existing browser answer
keys and original case/media bindings are retained.

Reviewers choose supported, conflicting, or uncertain, can copy and correct the
GPT proposal, and export UTF-8 CSV for one or all datasets. Drafts are exported
with `human_confirmed=false`. Answers remain in that reviewer's browser.
Import accepts the site's human-review export schema, validates all case IDs,
sequences and initialization frames before writing, retains newer local
records, and separates reviewer IDs. This static site does not provide live
shared server storage. Return exported CSV to the project owner for consolidation.
Dataset cards, keyword search, pending/reviewed/model-uncertain filters and
numeric ranges allow people to divide the 1,765 VOT anchors.

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

`import_model_reviews.py --depthtrack <152-row CSV> --external <1845-row CSV>`
binds reviewed content to the existing manifests and validates original
category, attributes and initialization identities. It changes no image/video
path, tracking input, checkpoint, or experiment metric.
