# M44 tracker — training complete, recursive validation running

| Stage | Status | Evidence |
|---|---|---|
| Architecture and comparison | Frozen | Geometry primary / coordinates-zeroed matched control;448,739 parameters each |
| Train sampling | Complete |85 sequences;2,101 pairs;1,511 fit /590 development;525 transition-tagged pairs |
| Set/geometry/optimizer contracts | PASS | Nontrivial permutation test; exact masked-geometry invariance;24 synthetic optimizer steps |
| Native collection smoke | PASS |120 frames, six consecutive pairs; exact default bbox/confidence; two real template writes |
| Full collection | Complete |85 sequences;126,382 tracked frames;2,101 pairs; exact default bbox/confidence;1,420 template writes |
| Training/runtime source | Frozen before real optimization | Binding390b236fe7fc5cca318e774ed5f05f4dd03d56b21f741ab668e64a2c71b353d3 |
| Runtime integration | PASS |120-frame default bbox/score/template/query parity; two forced-candidate frames and preceding selected-index propagation |
| Real fitting | Complete |20 epochs /960 updates per arm; final checkpoints strictly reloaded;46 tensors changed per arm |
| Static development | Diagnostic only |590 windows: default meanIoU.440274; geometry.441423; appearance.438411; primary is unchanged |
| Pipeline controller | Running recursive validation | PID13654; completed collection, runtime checks and both training arms |
| Full recursive development | Running | Started13:32:07; geometryPID15301/appearancePID15302; each22 complete sequences/33,130 frames |
| Terminal scalar/weight audit | Prepared, not executed | tools/audit_sttrack_m44.py runs after original controller exit0 |
| Public evaluation | Not launched | Requires the original main recursive gate; no automatic public launch |

Recursive performance remains pending until both terminal receipts exist. Do not
restart an existing collector because a monitoring call times out. Static
snapshots are diagnostic; the geometry primary and performance gates are
fixed before these results. No same-weight three-dataset result is claimed.

Both final weights require the same official STTrack backbone checkpoint:

- Geometry: `9ef0caf6e6a3a09d46771e9ea3d7073cfc0c0195f66979051220da97e517d978`.
- Appearance: `623e9a7e9f9c912e2ccb5e70238dde2805380d54519dd8adea8b930dddb2d07c`.
- Full training result: `7f6df58956ea6a9ad1cf018fb9763d26c14137bb412e11f89292ba60e78a5266`.

The primary's three new severe static regressions are all egg windows; the
correct default candidate exists in both frames. Correspondence-column
inspection gives unmatched/correct/wrong, so a correct matcher followed by an
incorrect final ranking is not a uniform explanation. See the sealed static
diagnosis; no inference policy or training setting was changed.
