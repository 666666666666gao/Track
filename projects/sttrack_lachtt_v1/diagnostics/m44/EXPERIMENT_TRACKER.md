# M44 tracker — frozen collection and pipeline launch

| Stage | Status | Evidence |
|---|---|---|
| Architecture and comparison | Frozen | Geometry primary / coordinates-zeroed matched control;448,739 parameters each |
| Train sampling | Complete |85 sequences;2,101 pairs;1,511 fit /590 development;525 transition-tagged pairs |
| Set/geometry/optimizer contracts | PASS | Nontrivial permutation test; exact masked-geometry invariance;24 synthetic optimizer steps |
| Native collection smoke | PASS |120 frames, six consecutive pairs; exact default bbox/confidence; two real template writes |
| Full collection | Running | PIDs13173/13175; two3090 GPUs;126,382 planned tracked frames |
| Training/runtime source | Frozen before real optimization | Binding390b236fe7fc5cca318e774ed5f05f4dd03d56b21f741ab668e64a2c71b353d3 |
| Pipeline controller | Waiting for original collectors | PID13654;240-second polling; then runtime parity, fixed fitting and recursive validation |
| Real fitting | Not yet started |20 epochs /960 updates per arm specified; synthetic contracts are not real training |
| Recursive / public evaluation | Not yet run |22 complete development sequences will follow training; no automatic public launch |

The launch record is provisional until actual terminal receipts exist. Do not
restart an existing collector because a monitoring call times out. Static
snapshots are diagnostic; the geometry primary and performance gates are
fixed before these results. No same-weight three-dataset result is claimed.
