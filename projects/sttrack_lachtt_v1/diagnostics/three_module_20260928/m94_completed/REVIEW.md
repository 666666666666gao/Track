# M93/M94 retrospective evidence review

Reviewer: independent-context Codex agent `/root/m94_evidence_review`, 2026-09-28.
`review_independence: same-family`; `acceptance_status: provisional`.
This is not cross-family acceptance or authorization to train or deploy.

**PASS for descriptive result accounting, with the claim limits below.**
No blocking arithmetic, answer-remapping, or prompt-leakage defect was found
in the completed diagnostics. This review did not open the private M92 answer
manifest, derive its per-event mapping, rerun Qwen, or alter completed results.

- M93 recomputation confirms 11/12 positive contrasts, mean 1.1041667 and
  median 1.125; stored deltas are exact and all scores finite. The primary
  conditions keep the crop and question fixed. These are uncalibrated
  category-response contrasts, not semantic truth or tracking performance.
- M94 contains 24 unique events and 96 finite A/B/N/U readouts. Every stored
  choice equals the restricted-token argmax and is A, with no argmax ties;
  softmax discrepancies from independent recomputation are below 1e-7.
  Swap consistency is 0/24 in each text condition. The code maps reversed
  A/B answers correctly; always-A choices imply zero both-order correctness
  when exactly one candidate is positive. The reported 12/24 single-order
  counts rely on the source's balanced assignment and aggregate receipt;
  hidden per-event labels were not independently inspected. Equal A-minus-B
  margins under swapping reproduce 16/24 and 20/24.
- `inputs_verified.json` confirms five image blocks for the first event:
  the first three match, the distinct candidate blocks swap exactly, and
  actual short replies are A in both orders. Its prompt matches the main
  visual-only prompt. This receipt covers one event and visual-only text,
  not all 96 unrestricted generations. Main scoring uses four selected
  next-token logits, as declared.
- Source inspection confirms raw RGB frames/crops and optional auto category
  enter Qwen; current GT, scores, sequence names, and answer mappings do not.
  Published M94 rows contain no per-event correctness, GT, or candidate index.
  The plan's earlier instruction to publish per-event correctness was fixed
  to aggregate correctness only. Reported source/result digests match files.

**Required limitations; these block stronger claims, not retention of the
descriptive measurements:**

1. M93 does not have equal effective masked pixel areas.
   `audit_region_evidence.py:49-56` redraws the target's two-pixel red border
   after masking; inclusive rectangle endpoints and corner clipping also
   differ. Executing the exact helpers on a blank 640x360 image gives target
   versus control painted-gray areas of 462/625 pixels for 013 and 702/870
   for 007; all 12 differ. Describe nominally equal box extents and this
   asymmetry, not an area-matched causal control. The existing measurements
   must not be silently replaced by changed-code results.
2. The M92 manifest file is private, but its correct-side assignment is
   reconstructible from the committed generator's fixed seed and sequential
   audit IDs (`prepare_candidate_review.py:83-94`). This is not leakage into
   Qwen's prompt, but the packet is not answer-private to a reader of that
   source. Future blind human review needs a newly private assignment; merely
   omitting the manifest does not restore blindness. No mapping is disclosed
   in this review.

The selected Train events, unverified categories, RGB-only inputs, and
unevaluated neither-valid/undecidable calibration prevent broader claims.
M94 fails its stated teacher gate; it supplies no online teacher or semantic
training labels. Independent human phrase and candidate review remains open.

## Corrective preparer addendum

2026-09-28, same reviewer and provisional same-family attribution.
**Static PASS: `reblind_candidate_review.py`; no blocking findings.**
The preparer was not executed and no private manifest or answer mapping was
read or derived during review. Syntax was checked with `ast.parse` only.

The script shuffles the same 24 in-memory rows using `SystemRandom`, assigns
exactly 12 positive-A positions with a separate unseeded draw, and renumbers
the presentation. Box selection, A/B indices, overlays, and unmarked crops
agree with the original generator's native-index-zero invariant. The new
manifest alone contains the old/new ID link and answer indices; stdout gives
aggregate counts only. Requiring a nonexistent output directory and writing
only beneath it preserves the source manifest, images, and completed results.
No training seed, tracker state, or weight is changed.

This fixes reconstruction from the new generator and presentation IDs alone.
It does not erase the old packet or make an already answer-informed reviewer
blind; independent human review must still respect that exposure boundary.
