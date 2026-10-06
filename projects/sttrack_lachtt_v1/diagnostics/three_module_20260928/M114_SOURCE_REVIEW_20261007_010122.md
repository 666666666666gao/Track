# M114 source review

Checked at: 2026-10-07T01:01:22.497950+08:00

Verdict: **PASS**. No remaining blocking or nonblocking issue. Fresh-context gpt-6-astra / max; same-family and provisional.

The reviewed change reads both completed M113 finals and measures candidate score responses without optimization. Full scope is 2544 fit and 495 development states per arm under Empty/object/human-text conditions. It neither commits a recursive tracker action nor supplies any of the nine official metrics.

The final sources implement the plan correctly. They strictly load the recorded checkpoint, disable gradients, use eval/no_grad, check finite outputs and exact Empty identity, check quality invariance, and compare every parameter and buffer with the loaded state. Checkpoint and bank/label digests bind the run to M113. The reused input path excludes GT IoUs from model inputs.

GPU0 uses the human final; GPU1 uses the generic final. Both 128-state sanity runs must pass before full starts. Child exits, status, zero optimizer steps, replay scope, exact counts and invariants are checked. Failure retains logs; there is no retry, training, seed search or promotion path.

Each arm emits 384 sanity rows and 9117 full rows. Stored replay covers own-arm fit and all three development conditions: 256 comparisons in sanity and 4029 in full per arm. The initial sanity receipt overstated replay scope. This was corrected before the final verdict: stored_selection_replay_matches and explicit first_batch_per_split/all_cached_states metadata are now emitted and checked.

CPU arithmetic is correct. Per-state score deltas split into a common mean and centered candidate-specific component; their energies add to total energy. Localization BCE uses each candidate's actual IoU target and divides by candidate count. Good/poor margins compare maximum scores at IoU >= 0.5 and IoU <= 0.1, and missing classes are counted. Poor localization is never labeled as a different physical identity.

Actual local evidence was checked:

- All 152 initialization records and 704 retained phrases match the human-confirmed CSV. Sequence partition is 130 fit / 22 development.
- Persisted GT-label bytes match preparation/download digests. Their 2544/495 valid-GT key sets exactly match every M113 event file.
- Both M113 sanity runs contain 3 updates and both finals contain 480. All exits are zero, every saved event/subgroup summary recounts exactly, and all inspected completed artifacts match M113_DOCUMENT_SYNC.
- Human-final development has 5 changed selections, 1 IoU improvement, 0 worsenings and mean IoU delta +2.984461760280108e-6 versus its own Empty.
- Native-poor/oracle-good events prove 68 fit and 16 development states contain both margin classes. Whole-split analysis therefore has eligible states without adding speculative fallback logic.
- Actual analyzer functions pass bounded stdlib synthetic checks for stable BCE, common shifts, candidate-specific shifts, margin crossings, absent-class counts and energy identity. No neural network was executed.

The private transport uses port 43811 and stdin-only password input. It verifies the accepted sources and its own digest, requires terminal M113 and absent M114 outputs plus idle GPUs, checks reused remote sources/uploaded bytes, and launches the bounded controller. This review did not read credentials or perform a network action.

The JSON companion records all 41 diagnostic-directory files actually read, both non-D provenance inputs, the private transport digest, checkpoint receipt digests, executed checks and limitations. Remote checkpoint/bank/feature bytes and GPU numerical replay were not evaluated locally; runtime must establish those results. The nine-metric project goal remains open.
