# Formal evaluation contract after a recursive pass

This is a read-only interface review made while M46's fixed recursive run is
active. No public tracker execution or new benchmark score is recorded here.

The accepted model must be recorded as one bundle: the official STTrack base
checkpoint, the accepted learned association checkpoint, the exact YAML and
runtime source hashes. All three datasets must use that same bundle. A training
head checkpoint alone is insufficient to identify the complete model.

| Stage | Dataset / protocol | Advancement condition |
|---|---|---|
| Train recursion | Existing22development sequences,33130frames | Original mean-IoU, low-frame, H10 and healthy-sequence gates |
| First public development evaluation | Frozen VOT low22,303multi-start anchors | EAO/ROB each at least1pp above M39default; ACC no more than.10pp lower; fewer failed anchors; protect all7 zero-failure sequences |
| Same-bundle complete evaluations | DepthTrack Test50, CDTB80, VOT-RGBD2022full127 | Only after the fixed low22 improvement condition |

Low22 has repeatedly served development. It must not be described as an unseen
test. Train H10 episode counts must not be converted into VOT ROB or failure
anchor counts.

The existing M39 VOT entry already uses the correct `rgbcolormap`,
`depth_clip=True`, search256/factor4, template128/factor2, two-template50/.75
update policy and fixed query window. Its verified TraX bridge is
`m39_vot_bridge.py`, SHA
`230acf10f378a6babfacf9979ea07a1ce89c34952cc0c6b9568376249e265316`.
The future entry must instantiate the accepted `STTrackCandidateSet` and its
association checkpoint while preserving this input/reporting path. Toolkit
analysis uses the existing mplt environment; tracking uses the sttrack
environment. Initialization is reset independently at each native VOT anchor.

For DepthTrack/CDTB, use ordered paired RGB/depth files and initialize from only
the permitted first-frame box. Later GT must be opened by analysis after the
predictions are sealed. The existing SRTrack dataset classes automatically
attach old language descriptions; they are not an appropriate direct entry for
the current language-OFF STTrack runtime. Reuse the validated metric routine,
with a thin native STTrack inference entry and the accepted bundle.

Keep the established report convention: six-decimal boxes and confidence
files, initialization confidence1, and the actual chosen candidate's native
confidence thereafter. Internal recursive state remains unrounded. Do not
change confidence normalization after viewing PR results.

The existing `depthtrack_pr.py` at `/home/SRTrack_RGBD_L/lib/test/analysis/`
has SHA
`05879f2e732aed982fbcbebd9756ce063ed0fa945c1f6b0c04092c3e487466cc`.
It uses bounded VOT region overlap, the existing confidence threshold grid,
per-sequence precision/recall curves followed by macro averaging, and the
resulting best F-score. Preserve the established initialization and absent-GT
handling. These PR/F-score values are distinct from the unbounded scalar
Train mean IoU used in the M44--M46 diagnostic gate.

Known normalized dataset roots are `/root/autodl-tmp/depthtrack/test/sequences`
and `/root/autodl-tmp/CDTB/sequences`; directory inventories are50/80 sequences.
The current complete counts are76373/101956frames. VOT's frozen303-anchor and
1765-anchor manifests, rather than a new selection of convenient anchors,
remain the evaluation inputs. No newly trained STTrack result may be combined
with historical SRTrack/SUTrack scores to claim a single-model three-dataset
achievement.
