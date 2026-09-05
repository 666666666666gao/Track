# M44 tracker — complete, primary performance failed

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
| Pipeline controller | Complete | Original controller exit0; original tracking processes ended |
| Full recursive development | Complete; both FAIL | All22 sequences/33,130 frames per arm; geometry.617098 vsdefault.652226; lowframes8436 vs7397; episodes87 vs75 |
| Terminal scalar/weight audit | PASS | All44 trajectory/source/weight bindings, scalarIoU/H10 and exact default prefixes verified |
| Public evaluation | Not authorized by result | Both performance gates failed; no low22/full127 launch |

Both recursive results and the terminal audit are sealed. Geometry is2.344925pp
worse than the appearance control, so explicit-position attribution also fails.
Appearance meanIoU.640547,7466lowframes,85episodes also fails and introduces a
failure on mobilephone02. No primary switch or public promotion is made.

Both final weights require the same official STTrack backbone checkpoint:

- Geometry: `9ef0caf6e6a3a09d46771e9ea3d7073cfc0c0195f66979051220da97e517d978`.
- Appearance: `623e9a7e9f9c912e2ccb5e70238dde2805380d54519dd8adea8b930dddb2d07c`.
- Full training result: `7f6df58956ea6a9ad1cf018fb9763d26c14137bb412e11f89292ba60e78a5266`.

The primary's three new severe static regressions are all egg windows; the
correct default candidate exists in both frames. Correspondence-column
inspection gives unmatched/correct/wrong, so a correct matcher followed by an
incorrect final ranking is not a uniform explanation. See the sealed static
diagnosis; no inference policy or training setting was changed.

The two largest geometry regressions were replayed to their first choice,
118frames exactly matching sealed boxes/scores. Eggframe16 chooses an incorrect
neighbor despite a correct candidate. Glass03frame102 has no IoU>=.5candidate,
yet commits candidate9; this single override changes the later trajectory.
See `first_choice_diagnosis.json`, `ASSOCIATION_DIAGNOSIS.md`, and the complete
`per_sequence.csv`. M45 is a separate fresh training-target intervention, not a
continuation or relabeling of this failed M44 result.
