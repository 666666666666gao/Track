# M99 Empty layout defect and bounded repair

The original final train stopped at its strict selection-logit versus visual-logit assertion, before an epoch completed. Original logs, exits and source remain in `m99_failed_initial/`.

The actual GPU diagnostic reset the same seed2027 model, optimizer and first-epoch fit order. The original expanded text layout failed in batch3 after three updates, with finite forward outputs and finite preceding losses/gradients/parameters. Score delta was 4.76837158203125e-7; affected entries were candlecup_indoor@634/candidate0 and duck04_wild@515/candidate4. The same encoded values made contiguous at the caller completed40 updates with exact zero semantic, phrase and score deltas. The temporary diagnostic saved no checkpoint and evaluated no development or public data.

This isolates the input layout at the shared text projection as the measured difference. The precise CUDA kernel responsible was not traced. Neither nonfinite arithmetic nor malformed text values were observed in this replay.

The proposed production fix is one line in shared `phrase_branch`: project `text.contiguous()`. It applies equally to full and null branches, preserves values and gradients, and adds no hard Empty bypass, fallback or relaxed assertion. The producer, data order, losses, frozen parameters, split, seed and final12epoch/480step budget stay unchanged.

Before corrected training, use `check_empty_replay.py` against original source to catch the actual expanded-input symptom, then against the fixed source to verify exact cancellation for all40 first-epoch batches. Run the existing GPU sanity and only then a fresh fixed-budget train in a distinct corrected output directory. The failed original has no checkpoint to resume. Production regression and new training results are not yet claimed by this preparation record.

All measurements here concern Empty numerical/function identity. They establish neither useful nonempty semantics nor recursive/public tracking performance. Independent semantic evidence gates remain in force.
