# Centered adapter real entry preflight

Scope: deployment correctness only, not a new training experiment or promotion.
Reuse sealed M84 final (c63605ebcb1f702de66f96967255e5301bfdca1e3c40450b6d4b8a952791b1e0), frozen M84 code and exact historical fit category/empty vectors. No M87 runtime changes, captions, external benchmark frames, subsequent GT, optimizer steps, or performance metrics.

Use the first 102 frames each of the first two frozen fit sequences: cube04_indoor and bag04_indoor. This is a predetermined source-order choice, not a result-based selection. Only initialization boxes are read from the training manifest. GPU 1 only while M87 trains on GPU 0. Expected duration under 10 minutes; do not occupy GPU 1 when M87 evaluation begins.

For Category and Empty, independently construct a direct tracker using the source sequence-indexed bank and native configuration. Compare this to the unchanged OPE entry using the observation-keyed bank and to the real toolkit/TraX subprocess transport. OPE reuses its tracker across two sequences; TraX uses a fresh subprocess per initialization, exercising reset behavior. Also run independent native tracking on both prefixes and require exact Empty direct bbox/score equality. No simulated transport may count as a real TraX result.

Expected calls: 2 conditions x 2 sequences x 101 transitions x 3 paths + 2 native prefixes x 101 = 1414 tracking calls, 14 initializations. OPE six-decimal error <=5.01e-7; TraX bbox must exactly equal float32 -> four-decimal text -> float32 conversion, confidence error <=1e-6. Require all child processes exit zero, all receipts complete, and all comparisons pass before claiming interface correctness. No inference about EAO/F/ROB follows from this check.

The copied initialization_text.vot_wire_bbox helper omitted the four-decimal serialization step established by prior real TraX tests. Fix only this unused-in-entry preparation helper in the isolated copy. Verify both offline fractional conversion and a real fractional initialization round trip using a separately bound key. This extra protocol-only session initializes the real tracker and sends no tracking frames. Count its initialization separately (15 total), never as performance.

Fresh same-family code review required by experiment-bridge before deployment. Existing M84 entry sources otherwise remain unchanged. Final M87 deployment requires its own final weight and input binding after its development result; this preflight does not authorize formal evaluation.
