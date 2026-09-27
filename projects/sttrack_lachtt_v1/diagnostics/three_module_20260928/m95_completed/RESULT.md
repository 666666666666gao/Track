# M95 initial-reference origin diagnostic — complete

Two RTX3090 shards collected 77/75 legal DepthTrack Train initializations.
All 152 auxiliary t0 reads left frame0 box/query unchanged and did not commit
the auxiliary query. The first real tracking frame reproduced the saved
native bbox and score exactly (maximum errors both0). The collector read no
subsequent GT or semantic labels. One-sequence smoke and both full shards
exited0; shard collection elapsed 11.1/9.9seconds, plus startup and CPU audit.
Only small initial-reference tensors were added; the M90 candidates are fixed.

Each reference and current candidate is mean-pooled over 16 RoI samples.
Selection uses the average RGB/Depth cosine, with no fitting or state action.
The first-use-template control exactly reproduces the earlier M90 diagnostic.

| Reference | Train scope | Valid events | Native correct | Cosine correct | Top10 oracle | Rescues / breaks | Native / cosine mean IoU |
|---|---|---:|---:|---:|---:|---:|---|
| First-use template | Former-fit130 | 2544 | 1575 | 78 | 1814 | 23 /1520 | .600203 /.142205 |
| t0 template | Former-fit130 | 2544 | 1575 | 84 | 1814 | 25 /1516 | .600203 /.142719 |
| t0 search | Former-fit130 | 2544 | 1575 | 1144 | 1814 | 28 /459 | .600203 /.426943 |
| First-use template | Former-development22 | 495 | 268 | 8 | 305 | 2 /262 | .524771 /.120183 |
| t0 template | Former-development22 | 495 | 268 | 9 | 305 | 2 /261 | .524771 /.120771 |
| t0 search | Former-development22 | 495 | 268 | 178 | 305 | 3 /93 | .524771 /.356466 |

Both template references are weak under this simple search-feature matching
rule; changing only the template encoding time changes little. Using an
initial **search** representation changes choices substantially, but still
loses90 native-correct development choices net. In the264 healthy development
events it selects173 correct and breaks91. Direct cosine must not replace the
native peak. This does not prove a particular positional encoding or TSG
operation causes the mismatch, or establish physical identity discrimination.

The result justifies one bounded, same-budget learned visual-reference
comparison (M96 first-use-template versus t0-search); it supplies no semantic
labels, recursive gains, or official P/R/F or EAO/ACC/ROB. Fit/development
counts come from the deliberately selected M90 event panel, not unbiased
frame sampling. The former22 are repeatedly used development evidence.

Sources: `collect_initial_feature_origins.py` and
`audit_initial_feature_origins.py`; predeployment review in `M95_REVIEW.md` is
same-family/provisional. Reports are `smoke_shard0.json`, `shard0.json`,
`shard1.json`, and `cosine_comparison.json`. Original reference tensors remain
at `/root/autodl-tmp/sttrack_m95_initial_origins_20260928/`.
