# M78: raw-response competition, fixed seed2027

Question: Does M77 degradation arise from added competition generally or its Hann-weighted formulation? Reuse sealed M73 (no added competition) and M77 (Hann competition); train only M78 RawCategory and RawEmpty. The same 289154-parameter adapter, initialization, 130 fit sequences, five-slot text banks, native frozen base, optimizer and 186694 tracking calls / 5798 updates per arm are retained. Both arms always use native Hann inference and default templates.

Only the added loss score changes: log(raw score), including raw-score ordering of at most nine eligible negatives. Rounded positive cell, detached outside-center / IoU<=0.1 negative mask and weight1 remain. Candidate membership can change with ranking; this is not an isolated change holding mined negatives fixed.

Ten prospective gates: Category pooled mean >= native by .002 and matched Empty by .001; macro mean >= both; low-IoU frames <= both; H10 episodes <= both; protect each reference's zero-H10 sequences. No accumulating historical successful-sequence union. Historical M65/M73/M77 results remain unchanged and are reported, not mandatory dominance criteria.

Final checkpoints only; no seed search or checkpoint selection. Run fixed-head Empty/Swapped diagnostics even after development failure. Concrete content claim requires original Category pooled/macro superiority, non-increased low frames and H10 versus both contents (eight conditions). Content success cannot override failed primary gates. No automatic public evaluations.

If both ranking versions degrade relative to M73, extra ranking is not supported under this budget. If raw improves over Hann, this supports a formulation-specific issue, not removing inference Hann. If original words underperform same-head alternatives, do not claim useful semantic content. Local ranking cannot repair absent-crop observations.

Budget: two GPUs, approximately 4.5 hours training plus 35 minutes development; content controls about 35 minutes. Poll near estimated completion, 240-second cadence when needed. Preserve both Qwen models. No new weights downloaded, smoke weights not saved. Independent requested reviewer is unavailable; deterministic checks are not a reviewer PASS.
