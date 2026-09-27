# M94 candidate identity and order-swap pilot — failed gate

Qwen2.5-VL-3B-Instruct was run read-only on the **24 fixed M92 DepthTrack Train
fit130 hard events**. Every event has one candidate with Train-GT IoU≥0.6, a
different native candidate with IoU≤0.1, and mutual candidate overlap≤0.1.
The hidden correct side was balanced A/B=12/12 and never shown to Qwen.
These deliberately selected native-error states do not estimate the frequency
or performance of ordinary tracking frames.

Qwen saw the initial red-boxed RGB frame and target crop, the current
unannotated full RGB frame, and two candidate RGB crops. A/B crops were shown
in both orders. Each order was scored once with no category and once with the
existing **unverified automatic category**, using the first-token logits for
`A`, `B`, `N` (neither), and `U` (undecidable). The initial/current full frames
were unchanged across swaps. Qwen was not shown Depth, GT, model scores, or
sequence names.

| Input text | Correct in A/B order | Correct after swap | Both orders correct | Swap-consistent answers | A/B/N/U choices in each order |
|---|---:|---:|---:|---:|---|
| No category | 12/24 | 12/24 | **0/24** | **0/24** | **24/0/0/0** |
| Unverified auto category | 12/24 | 12/24 | **0/24** | **0/24** | **24/0/0/0** |

The predeclared teacher gate required at least 20/24 both-order correct and no
more than two swap inconsistencies. Both conditions fail decisively. Every
readout selected the first-listed crop, so 12/24 single-order accuracy is
exactly the balanced A-side frequency, not evidence of instance recognition.
The auto category changed some raw logits but none of the 48 order-specific
choices. The A-minus-B margin change under swapping was zero in 16/24 events
without text and 20/24 with auto text; this is exploratory score analysis,
not a corrected deployable predictor.

The earlier 9-event Qwen2.5-VL-7B free-answer audit also reported strong A
bias (A=7, B=0, abstain=2), but its event set, model, image protocol, and
labels differ. M94 specifically closes the previously missing A/B-swap test
for this new 3B RGB-only prompt. It does **not** show that all VLMs or all
prompts fail; it shows this protocol is unfit as an online identity teacher.
The packet always contains one IoU≥0.6 candidate, so `N`/`U` calibration on
neither-valid or genuinely undecidable states was not evaluated.

Use bfloat16, seed2027, `torch 2.5.1+cu121`, `transformers 4.51.3`, and
`qwen-vl-utils 0.0.11`. The one-event preflight and full24 run completed with
finite logits. Source `audit_candidate_vlm.py` SHA-256
`613446b7910f0c1c46e4ab374af2e109fbfc31ad0769f6acb832cfe0debcf07d`;
[`full24.json`](full24.json) SHA-256
`00d262d6d24ebb5fab96dc792fe4029216ce5daf3f7cd5ba8bdb3f4297736d0b`.
The JSON contains per-event choices/logits and aggregate correctness, but
**no per-event correct side, GT box, or hidden candidate mapping**. The
private M92 answer manifest remains outside GitHub. Remote run directory:
`/root/autodl-tmp/sttrack_m94_candidate_vlm_20260928`.

No tracker checkpoint, recursive state, benchmark result, or semantic training
label changed. Module A still needs independent first-frame phrase and
candidate identity review; this Qwen protocol must not provide its labels.
