# Full152 same-weight content controls

Use each final Full152 checkpoint, original external initialization observations, and the frozen OPE interface. Empty replaces only valid category slot 0 with the encoder's empty-text vector. Swapped replaces only slot 0 with a deterministic category from another observation whose category string differs. Both retain slots 1–4, mask, padding, RGB-D initialization, tracker state rules, and checkpoint. Each intervention begins a fresh trajectory from frame 0.

The swapped input is a lexical perturbation, not a verified false description. Category versus Empty or Swapped measures same-weight input sensitivity across complete recursive tracks; neither control is an independent visual-only trained model. M67/M82 Empty still passes through the learned additive adapter and does not restore native STTrack.

After the primary Full152 VOT evaluation finishes, `run_ope_controls.sh` prepares the eight plans and runs DepthTrack Test50 and CDTB80 in parallel on GPU 0 and 1 for each model and variant. It reuses the frozen `run_semantic_ope.py` tracker and analyzer. Keep these diagnostic predictions separate from the primary evaluation root. The script is prepared but is not part of the current VOT process.
