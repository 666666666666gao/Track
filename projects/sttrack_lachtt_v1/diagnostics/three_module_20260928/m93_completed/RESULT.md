# M93 initialization region-evidence pilot — complete

This is a **read-only Qwen diagnostic**, not STTrack training or an official
tracking result. It uses 12 DepthTrack Train fit130 first frames selected from
the earlier 13-case assistant-screening pilot. `bike01_wild` (`021`) is excluded
because its 297×246 target box has no disjoint same-size corner region in a
640×360 frame. No human semantic review was complete at the time of this run.

The unchanged automatically generated category was inserted into a one-word
Yes/No question about the red-boxed target and its crop. The primary paired
comparison keeps the target crop and question identical, masking either the
full-frame target or an equally sized unrelated corner region. `target_both`
masks both the full-frame target and the crop and is an additional, unpaired
sensitivity readout. Values below are `Yes` minus `No` first-token logits;
the final column is `unrelated_full - target_full`, positive when masking the
target lowers category evidence more than masking the control region.

| Audit | Auto category | Original | Target full masked | Unrelated full masked | Target both masked | Paired difference |
|---|---|---:|---:|---:|---:|---:|
| 002 | case | 2.500 | 1.000 | 2.375 | 0.125 | +1.375 |
| 005 | billiard ball | 2.500 | 0.750 | 2.500 | -0.625 | +1.750 |
| 006 | apple | 1.750 | -0.750 | 1.750 | -3.750 | +2.500 |
| 007 | ball | 2.625 | 0.750 | 2.625 | -1.500 | +1.875 |
| 010 | basketball | 2.375 | 0.750 | 2.375 | 0.625 | +1.625 |
| 011 | basketball | 1.000 | 1.125 | 0.750 | 0.750 | **-0.375** |
| 012 | soccer ball | 2.125 | 1.250 | 2.250 | -0.375 | +1.000 |
| 013 | cup | 0.250 | 0.250 | 0.500 | 0.000 | +0.250 |
| 023 | bike | 2.250 | 1.375 | 2.250 | 0.875 | +0.875 |
| 024 | book | 1.750 | 0.625 | 1.875 | -0.750 | +1.250 |
| 026 | book | 1.625 | 1.375 | 1.750 | 1.000 | +0.375 |
| 029 | bottle | 1.625 | 0.625 | 1.375 | 0.250 | +0.750 |

The paired difference is positive for **11/12** cases, negative for **1/12**;
its mean is **+1.1042** and median **+1.125** logits. The one negative case,
`011`, had an assistant-only preliminary judgment that the generated
`basketball` label conflicts with the visible American football. Conversely,
`006` (`apple`) was assistant-screened as uncertain yet has the largest
positive difference. These examples show why region sensitivity is not
semantic correctness. The assistant judgments are not independent human truth.

The original one-case preflight used an unbalanced three-condition design. A
first 13-case attempt was stopped by the explicit nonoverlap assertion at
`021` and produced no complete result. The balanced 12-case protocol above
was fixed before the successful run; its 12/12 records have finite logits and
verified first-image SHA-256 bindings.

Model: local Qwen2.5-VL-3B-Instruct, bfloat16, SDPA, seed2027; runtime
`torch 2.5.1+cu121`, `transformers 4.51.3`, `qwen-vl-utils 0.0.11`.
`config.json` SHA-256 `7ed3eed5be6924cc800e8a5e53fc405c1aab1aaf36bad65c33403b36c56827f5`;
model index SHA-256 `c7dd78a4c6bea60b51332f1baf37b8f8124ecab2c35395a29a29825bf2619768`.
Source SHA-256 `08f325a61b3fad7d4e6b9c185a652b7c18d8960d3037ac805ca085ec281f300a`;
[`balanced12.json`](balanced12.json) SHA-256
`998b58e853e1d41b40131ab5e635e8fcdd346e34bcd896f71ec4e97d500fa97f`.

Gray masking shifts the input distribution. The Yes/No logit is an
uncalibrated VLM response, not an instance identity score. This selected
12-case pilot provides no tracking P/R/F, EAO/ACC/ROB, or evidence to promote
Module A+B. The next semantic training gate remains independent first-frame
and same-instance candidate review, followed by correct/Empty/conflict
fixed-state comparisons on Train data.
