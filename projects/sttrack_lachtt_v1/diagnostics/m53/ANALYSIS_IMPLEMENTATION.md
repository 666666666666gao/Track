# M53 post-seal capacity analysis implementation

The analyzer was frozen separately from the running collector. The collector,
spec, controller, base checkpoint and predeclared capacity screen are unchanged.
`analysis_binding.json` binds the new source, execution binding, native training
binding and synthetic checker before opening experiment GT for this analysis.

`analyze_sttrack_m53.py` requires both collection exits to be zero and checks
all 63 output hashes, sizes, frame coverage, native-output agreement, past-view
ordering and replay/state-preservation assertions before any GT file is opened.
GT frame indexing follows the original zero-based event manifest. As in the
native recursive analysis, the dataset GT is restricted to the bound sequence
length; invalid GT does not contribute to overlap means or capacity counts.

The three modes keep the current-template control available:

| Mode | Historical alternatives | Selection |
| --- | --- | --- |
| default | None | Native current-template top1/top10 |
| all_past | All strictly past native writes | Privileged best view |
| valid_past | Past writes with write-frame GT IoU >= 0.5 | Privileged write filtering and best view |

For each valid event, the output retains every view's top1/top10 overlap and
write-time overlap, with separate template frame IDs for the best top1 and
top10 views. These can be different views. Reports distinguish improvement of
ranking within existing current-template candidate capacity from newly available
top10 capacity. They also include current-overlap strata and all 63 sequences.

Because retaining the current control makes oracle means non-decreasing by
construction, individual historical-view harms are counted separately: a
current-template top1 IoU >= 0.5 becoming <= 0.1 under an alternative. Both the
all-past and privileged-valid-past subsets report exposed read counts and harmed
event counts. The oracle result alone cannot establish safe template selection.

The original screen remains unchanged: at least ten severe current events
rescued by a past-view top1, across at least three fitting sequences. It applies
to all past alternatives; the valid-past subset is an additional diagnostic.
Passing permits reader design, not tracker promotion or automatic training.

The hand-computed synthetic checker covers separate top1/top10 view choices,
historical-write filtering, current-control retention, individual healthy-read
harm, invalid-GT exclusion, rank-versus-new capacity and finite JSON output.
It passes locally through `uv` and in the server's STTrack Python 3.8 environment.
`analysis_contract.json`, `.exit` and `.log` are the server execution evidence.
No experiment GT is read by that checker. The independent reviewer found no
concrete code defect; their local bare-Python checker attempt stopped at a
broken `python.exe` shim before executing code. That attempt is retained in the
verbatim review, separately from the successful server check.

No full capacity analysis result, reader weight, recursive improvement or public
benchmark result is claimed by this implementation note. The full collection
must finish and pass its receipt checks before the analyzer runs.
