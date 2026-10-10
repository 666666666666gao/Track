# M122 Current-quality C Full152: a new final-fit/evaluation branch

Status: source prepared locally; no training, deployment or official results yet.

The original Future-C development gate failed. This branch does not execute,
replace or retrospectively pass the original Future-C R4 plan. It takes the
Current-quality C control forward because its 32-sequence recursive development
run improved equal-sequence IoU and H10 relative to frozen P1 with the native
template rule. Its effect was concentrated in three sequences; 29 were unchanged.
These are development observations, not new official benchmark results.

## Fixed training and model

- Reuse the frozen P1 A+B final (SHA256
  `3fd4087ca0e9f892ce2b2a0647fa1b706bfc7e869099f6c24c129225ede96811`).
- Use the existing audited original-P1 teacher covering all 152 DepthTrack
  Train sequences and 219,802 tracking calls. Load the 1,257 common eligible
  events (993 former fit + 264 former C development), spanning 142 sequences.
  Preserve the 35 current-GT-unknown and 5 future-unavailable exclusions.
- The former development32 are now training data. No subsequent result on
  those sequences is claimed as unseen C validation. A+B already saw all152.
- Train only the existing 70,721-parameter C MLP from seed2027 initialization:
  519→128 ReLU→32 ReLU→1; current GT IoU regression; AdamW lr3e-5, wd.01,
  batch32, ten fixed epochs, exactly 400 updates, fixed epoch10 final.
- Save one composite `final.pt` containing the unchanged P1 A+B and fitted C.
  No external Test/CDTB/VOT frames or labels enter this optimization.
- C deployment remains native qualification every50 frames with native
  response at the selected cell >.75, followed by C>.5. Retain permanent initial
  identity, exact selected box/query/score interface, and the same human-verified
  category plus up to four stable attributes. No language regeneration.
- This trains a current-quality control. An IoU quality head alone is not a new
  future-memory-utility contribution. Teacher labels describe original P1 states;
  they are not transferred as ground truth for changed C histories.

## Complete official evaluation

Use the same composite final, initialization bindings, human text, native weights
and inference policy on DepthTrack Test50/76,373 frames; CDTB80/101,956 frames;
VOT-RGBD2022 full127/1,765 legal anchors. Reuse sealed P1 cases and VOT protocols,
not their predicted trajectories. Preserve the locked OPE and official VOT metrics.

GPU0 fits C, then tracks DepthTrack; GPU1 tracks CDTB concurrently. Once both OPE
predictions are complete, the VOT shards run in two waves using both GPUs.
Metric-only stages use the existing mplt environment. Blocking child waits record
real exit codes; no controller polling loop, automatic retry, environment rebuild,
checkpoint search or per-dataset strategy selection is introduced.

Before deployment, complete the fresh R3 raw-result audit and source review of
this new branch. Preserve their actual limitations. The next GPU admission or
remote NN observation is not before 2026-10-10 22:19:36.400301 +08:00. Thereafter
observe at most hourly, with an actual resource snapshot at launch. Native local
long-wait joins are 300 seconds; they do not inspect training progress.

Acceptance requires one complete row: DepthTrack P/R/F≥65.2/64.9/65.1;
CDTB P/R/F≥72.9/75.6/74.2; VOT EAO/ACC/ROB>77.9/82.1/93.7. Training completion
or development gain does not pass this gate. Seal raw outputs and independently
audit the actual completed run before making performance claims. If targets are
unmet, retain every metric and analyze gains and harms before the next change.
