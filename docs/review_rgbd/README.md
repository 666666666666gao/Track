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
GPT-6 Astra Pro reviews have been imported for all 2,047 cases: DepthTrack
Train152, Test50, CDTB80, and VOT1,765 legal initialization points. Each case keeps its
original generated description alongside the model's proposed category,
initially supported attributes, uncertainty and separate later-frame notes.
The source files are hashed in `data/model_review_receipt.json`;
`data/gpt_initial_reviews.csv` is the normalized downloadable initial-review table.
DepthTrack Train and Test use the supplied 2026-10-06 GPT re-review and the user's
new 20261006 human round. All four groups now include 2,047 submitted human
records under reviewer `gb`; previous DepthTrack human decisions are not used.
The uploaded status values and confirmed categories are retained exactly. GPT opinions
are not promoted to human confirmation. Test50 has no supplied original caption,
so its review concerns the GPT proposal rather than an unavailable old caption.
Original case/media bindings are retained.

Reviewers choose supported, conflicting, or uncertain, can copy and correct the
GPT proposal, and export UTF-8 CSV for one or all datasets. Drafts are exported
with `human_confirmed=false`. Answers remain in that reviewer's browser.
Import accepts the site's human-review export schema, validates all case IDs,
sequences and initialization frames before writing, retains newer local
records, and separates reviewer IDs and review rounds. DepthTrack uses round
`20261006`; CDTB/VOT retain their original browser keys. A record imported without
`review_round` belongs to the original round and cannot confirm this DepthTrack
round. Submitted human records appear directly on the page; selecting `gb` loads
them for continued review. This static site does not provide live
shared server storage. Return exported CSV to the project owner for consolidation.
Dataset cards, keyword search, pending/reviewed/model-uncertain filters and
numeric ranges allow people to divide the 1,765 VOT anchors.

Only DepthTrack Train review may inform future training. CDTB and VOT review is
for input auditing and error analysis; its labels and later frames must not
enter training or model selection. Inference descriptions use only each legal
initialization observation.

The user's 2026-10-07 protocol requires human-confirmed initialization text for
future training and inference. Train's confirmed category and stable attributes
are training inputs; Test/CDTB/VOT records remain external initialization inputs.
Sequence-name prefixes are review clues and never silently replace the submitted
category. `import_confirmed_human_reviews.py` imports the submitted CSVs while
preserving the original GPT proposals and media. Source records and coverage are
in `HUMAN_REVIEW_IMPORT_20261007.json`.

`build_assets.py` reads the frozen bfloat16 CDTB/VOT generation plans and
records, verifies the record-file hash and each image binding, then makes
boards, manifests, and review videos. `build_depthtrack_assets.py` uses the
existing DepthTrack Train initialization manifest and creates the same format.
External later frames are unmarked review context. Train152 additionally has
nine-frame target crops located using training GT, explicitly labeled as offline
review evidence. Test50 adds unmarked nine-frame sequence overviews, eleven
original JPEGs per case, and 2fps sampled whole-sequence previews. These later
frames do not become legal initialization information.

`build_clear_assets.py` regenerates the clearer evidence for all three datasets.
It retains existing preview paths and descriptions, verifies original
initialization-image hashes and source-manifest bindings, and writes versioned
detail/video files plus build receipts. Later frames remain unmarked review
evidence and do not enter external-dataset training.

`import_model_reviews.py --depthtrack <152-row CSV> --external <1845-row CSV>`
binds reviewed content to the existing manifests and validates original
category, attributes and initialization identities. It changes no image/video
path, tracking input, checkpoint, or experiment metric.

`import_review_round.py` binds the supplied Train152 CSV and Test50 ZIP CSV to
their original blank templates, verifies identity and source bytes, imports the
submitted CDTB/VOT human CSVs, and copies verified offline review materials.
`REVIEW_ROUND_IMPORT_20261006.json` records hashes, coverage, and round boundaries.
Re-running the same import over an already-imported round is intentionally rejected.
