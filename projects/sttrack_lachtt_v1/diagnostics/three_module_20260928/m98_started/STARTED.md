# M98 context/support completion started

Source review: fresh gpt-6-astra/max PASS, same-family/provisional.
One-case GPU0 smoke completed in10.54s with three cube04 Train events
[10,12,14],14native prefix calls and zero bbox/score error. Exact M90 event
boxes/regions and M95 t0 region assertions passed. The four initial fields
and four first-event fields also exactly reproduced M97's actual cube04
smoke tensors; see `smoke_comparison.json`. No GT, text or optimizer step.

Two full collectors started at **2026-09-28 08:36:15 CST**, in screen
`m98_context_collect_20260928`, with verified live Python PIDs259894/259895.
Their queue binds GPUs0/1 and preserves the original77/75sequence shard split.
Planned output is152Train sequences,3502existing events and219194native calls,
followed by a CPU audit only if both workers exit0. Initial CUDA usage was
still1MiB during import at the three-second launch observation; this receipt
proves the processes started, not that full collection or GPU work completed.

The post-start **08:38:46 CST** observation then confirmed actual GPU work:
both worker commands were live, each GPU used2444MiB at78%/68% utilization.
Shard1 completed bag04's23events/1664prefix calls with maximum bbox/score
errors0; shard0 was still on its first long sequence. The full queue had no
terminal exit. See `bootstrap_observation.json`. No additional routine progress
poll is scheduled before the hourly check.

Expected slower-shard completion is roughly10:36 CST based on the earlier
native collection; this is an estimate. Next regular status check is09:36 CST.
Remote output/log/exit files:
`/root/autodl-tmp/sttrack_m98_train_contexts_20260928/`.

Only missing surrounding contexts, sample-support masks and raw nonzero depth
fractions are stored. Candidate regions/boxes/score/geometry are reused from
M90, initial regions from M95. No training/checkpoint, labels, recursive action,
C change or official metric is represented by this start. Independent reviewed
semantic/instance labels remain pending for the semantic A+B training gate.
