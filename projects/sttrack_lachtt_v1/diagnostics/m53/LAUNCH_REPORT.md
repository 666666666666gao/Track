# M53 historical template read capacity: contract complete, collection running

The frozen plan keeps the native STTrack network, initial template, search,
query state, default template updates and recursive output. At the original
1,511 fitting events, each strictly past native template is read separately
in the dynamic-template slot. All past native writes are retained, so this
experiment measures capacity with an archive; it does not yet validate a
small memory, a causal reader, or a trained architecture.

| Completed purity contract | Observed |
|---|---:|
| Fitting sequences / tracking steps | 2 / 240 |
| Current-template replay events | 238 |
| Historical template shadow forwards | 110 |
| Native template writes | 3 |
| Maximum native bbox / score error | 0 / 0 |
| Plain native outputs, templates and queries | Exact |
| Current-template score/size/offset replay | Exact |
| Public state after each shadow | Unchanged |
| Subsequent GT read / optimizer steps | None / 0 |

The first invocation stopped at import because the runtime tools directory
lacked the already published M41 helper. The exact bound helper was uploaded;
the collector and spec bytes did not change. The original failed invocation
is preserved as `contract_attempt1.log/.exit`. The subsequent contract exited
0, before any main collection or GT capacity analysis.

Main collection launched at 2026-09-06 00:08:44 CST in screen
`sttrack_m53_template_reads`. The 00:10 live-process check confirmed controller
125490 and Python 125491, GPU0 using 2,447 MiB, and no terminal exit files.
GPU0 held only 1 MiB immediately before launch. The independent native VOT
full127 reference remains on GPU1.

The sealed native predictions imply 14,349 past-template shadows and 109,222
total forward calls: 93,362 native tracking steps, 1,511 current-template
replays, and the historical shadows. These are predicted workload counts,
not a completed receipt. At 00:15, the first four sequences had completed
5,724 steps and 113 events with 436 historical shadows; their native bbox and
score errors were all zero. Their cumulative measured time was 316.379 s.
The initial total-duration estimate is about 90 minutes, near 01:40 CST;
check near the expected end and then use 180-300-second observation intervals.

Spec SHA256:
`d6f753537170f6f0a4d8dcfe3ec3b869af3e8fbb08ffc224a969fc9ab83450f2`.
Contract SHA256:
`41c3cf1a37b1262948665181f0351e410d15b663523ae9a34b8d896afec43c4b`.
Execution binding SHA256:
`a5e366d9960d04adada98529d19c57675d813ea88656cd06dfeafdce430e8668`.

The independent-context GPT-5.5 xhigh advisory review found no concrete defect
in the inspected plan, collector, contract or launch binding. It retains WARN
because complete collection, post-seal GT analysis, and capacity outcomes are
not yet available. The full response is in `EXPERIMENT_AUDIT.md`; this is a
GPT-family Type-A review, not external certification.

Next, verify all 63 completed files and source bindings before opening GT.
Report all-event and historically GT-valid-template capacity separately,
including harmful alternatives on healthy events. Best-view selection and
historical GT filtering are privileged diagnostics. The frozen 10-event /
3-sequence capacity screen is not a tracking promotion rule. No new weight,
causal reading policy, recursive gain or public benchmark result exists yet.
