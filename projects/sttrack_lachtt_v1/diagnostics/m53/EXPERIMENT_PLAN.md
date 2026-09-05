# M53 historical template read capacity, before execution

M52's equal-budget heads both failed the fixed recursive advancement gate.
M50 already showed that additional scale-triggered writes can harm recursion
even when most new crops overlap the target. This experiment asks a narrower
question before learning any template policy: does a previously observed
native template improve current prediction from the identical search/query
state?

Use the unchanged native STTrack base, initial template, factor-4 search,
fixed query window and default 50-step/confidence-above-0.75 update rule.
There is no candidate association head, new network, text, extra template
write, or changed recursive output. Archive every template the native path
actually writes, including initialization. At each of the original 1,511
fixed fitting events from the same 63 DepthTrack Train sequences, evaluate
each strictly past archived dynamic-template alternative. The initial input
template always stays fixed; the two-template network interface stays fixed.
The currently active dynamic template is the control, not an extra alternative.
All past views are included; this is capacity with an unbounded past native
write archive, not a claim for a deployed small memory bank.

The original event manifest was selected in earlier training diagnostics; it
is reused unchanged, not claimed to be a GT-free event trigger. The collector
itself reads images, permitted initialization boxes and sealed predictions,
never subsequent GT or labels. It follows the exact native trajectory through
the final event (93,362 steps), including healthy and failed sampling strata.
Archive writes made on frame t can only be read from frame t+1 onward.

Capture the actual pre-forward search, template and query inputs. Reuse M41's
query cloning and state hashing: STTrack forward updates the supplied query
list, so each shadow must receive its own clone. Replay the currently active
template as a same-state control and require identical score/size/offset maps.
Decode every historical view using its own size/offset maps; store its native
top1 box and Hann NMS top10. Historical outputs never alter the main bbox,
query or templates. Check the main path against its sealed native predictions
with the existing 1e-4-pixel/1e-6-score serialization tolerances.

Before collection, run two fitting sequences (chair01 and cube04), 120 steps
each, against an independent plain native tracker. Require exact outputs,
templates and queries, at least one native template write, at least one past
template shadow, identical current-template replay, and unchanged native
state after every shadow. This is a purity/interface contract, not a
performance test.

Seal all per-sequence outputs and source hashes before opening GT for analysis.
Report current-template and all-past-view top1/top10 capacity on the same
valid events, broken down by current overlap and sequence. Distinguish cases
where the current top10 already contains a valid target from cases where only
a past view creates one. Separately restrict archived writes to crops with
GT IoU >= 0.5 at their historical write time; that restriction and best-view
choice are explicitly privileged diagnostics, never deployment inputs.
Report harms from individual alternative views on healthy events as well.

Ten or more current-severely-wrong events with a past-view top1 IoU >= 0.5
across at least three fitting sequences would justify designing a reader
experiment. This is a predeclared capacity screen, not a tracker advancement
gate, and passing it does not demonstrate that a causal reader can identify
those views. Failure stops this archive-read version. There is no automatic
training or public evaluation. Any learned reader must subsequently be
trained on DepthTrack Train and pass the existing recursive and low22 gates
before same-bundle DepthTrack Test/CDTB/full-VOT evaluation.
