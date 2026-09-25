# M82 Full152 same-state Head readout

The sealed M82 Category predictions were replayed exactly for all traversed frames in four selected CDTB sequences. At 147 selected positions, the frozen fused feature was passed through the unadapted Head while the Category path alone continued to update tracker state. Both readout and posthoc metric processes exited 0. `analysis.json` SHA256 is `ad59402909a2bbf9225ac774a3b28b0d8078879c26176cb8c6bddeda996e1622`.

| Sequence | Selected frames / valid GT | Adapted vs native Hann peak differs | Adapted score + adapted geometry mean IoU | Adapted score + native geometry mean IoU | Severe frames in both |
| --- | ---: | ---: | ---: | ---: | ---: |
| `two_mugs` | 10 / 10 | 0, all peaks identical | 0.904452 | 0.911455 | 0 |
| `bottle_room_occ_1` | 46 / 10 | 18, all during invalid GT | 0.311865 | 0.320734 | 6 |
| `robot_corridor_occ_1` | 43 / 15 | 9, all during invalid GT | 0.113463 | 0.131332 | 11 |
| `jug` | 48 / 33 | 6, two with valid GT | 0 | 0 | 33 |

For these selected valid frames, substituting the native spatial score while keeping geometry fixed did not change the overlap or severe-error counts. Native geometry changed some overlap values but did not rescue any severe frame. At the first valid frame after the listed invalid intervals (`bottle` 574, `robot` 549, `jug` 286), all four Head combinations have IoU 0 in the already formed M82 crop.

During invalid-GT spans, adapted and native spatial peaks sometimes diverged substantially on the same state. At `bottle` frame 506 the Hann peak indices are 153 versus 137; at `robot` frame 457 they are 146 versus 121, producing widely separated readout boxes. GT is invalid there, so neither peak can be called correct from this evidence. These are single-frame uncommitted alternatives on Category history, not native STTrack trajectories or a causal intervention on future recovery. The cases were selected after observing external failures, and the readout does not attribute any difference specifically to word meaning.
