# M96 learned visual-reference control — complete, retain as diagnostic

Both visual arms use the M91 architecture, seed2027, identical candidate
features/geometry, fit130 IoU soft targets, epoch permutations, AdamW3e-4,
batch64, fixed12epochs, and final checkpoint. Only the initial reference is
changed. Fresh same-family/provisional review passed; one-epoch t0-search
sanity and both RTX3090 arms exited0. No text, physical-instance identity
labels, public benchmark data, recursive frame submission, or memory changes.

| Scope | Valid events | Native correct | First-use template | t0 search | Top10 oracle | First-use rescues/breaks | t0 rescues/breaks |
|---|---:|---:|---:|---:|---:|---:|---:|
| Former-fit130 | 2544 | 1575 | 1590 | 1587 | 1814 | 17/2 | 15/3 |
| Former-development22 | 495 | 268 | 270 | 271 | 305 | 3/1 | 4/1 |
| Development healthy | 264 | 264 | 264 | 264 | 264 | 0/0 | 0/0 |
| Development transition | 127 | 4 | 4 | 3 | 13 | 1/1 | 0/1 |

Development mean IoU is native.524771, first-use.528513, t0-search.527995.
Fit mean IoU is native.600203, first-use.602517, t0-search.601697. Thus the
t0-search arm adds one development correct choice, but its mean IoU is lower,
fit choice count is lower, and transition choice count falls4 to3. It passes
the plan's two minimum checks (more final choices than first-use and zero
healthy breaks), but does not provide sufficient evidence to promote this
visual prototype into recursion. The semantic A+B gate remains untested.

The first-use control exactly reproduces M91 visual metrics and all final
parameter tensors (maximum absolute parameter difference0). Its saved weight
SHA is unchanged. This fixed-state reproduction does not resolve the earlier
M89 recursive-training control divergence or establish general determinism.
Both495-row development reports were independently recounted for selected
hits, rescues and breaks. These related, deliberately selected events do not
estimate official metrics or represent independent trials.

Reports: `first_use_template/result.json` SHA
`51e500401e67ea7c1bfe9dc4a03ab63353d408162ddd229c63ed69ea2be9b81e`;
`t0_search/result.json` SHA
`21e1a9397079f4b7b7742a7ee34f1c64bbeae5d12f1fb88179d915d26393b843`.
Weight SHAs respectively
`69ccd4ad83740c60311980d6b192bc5514b1230fe531e5439d03f38516833ecb`
and `d6b04b043066a4a82c70329f8de7cb6bb44051e4b46ecebd6fecccc808cac9db`.
Weights remain at `/root/autodl-tmp/sttrack_m96_origin_control_20260928/`.
No additional visual training or public full evaluation is justified by this
comparison. Next A+B work needs independently reviewed Train phrase and
candidate-instance evidence; current private human review fields are blank.
