# M77 training aligned to final windowed peak competition

One user-required seed2027. Same M73 Category/Empty initialization, frozen visual base, 289154-parameter adapter, 130 fit sequences, text banks, optimizer and inference. The only change from M73 is the added training loss in window_competition.py; the only difference within the new pair is lexical content. No new inference head, gate, template rule or online caption.

The original focal and box losses remain. The added unit-weight loss compares log(Hann * score) at the native rounded GT cell against up to nine hardest decoded severe negatives. Negative candidates have centers outside the GT box and IoU <= 0.1; geometric masks are detached to exclude overlapping target hypotheses. GT is exposed only after the causal prediction/state commitment.

Native parity and real causal smoke checks passed before the frozen full training launch. They establish implementation contracts, not improved tracking performance. Smoke weights were not retained. Full training runs on GPU0 Empty and GPU1 Category; run_pair.sh then runs full Train development22 recursion and analysis. All run exits are individually checked.

The preparation script derives the isolated runtime from SHA-pinned M73 server artifacts. integration.json identifies the unchanged base source snapshot; no base/adapter checkpoint is distributed here. See EXPERIMENT_PLAN.md for protocol, limits and the 15 frozen development criteria. Same-head content diagnostics are predeclared regardless of primary gate status but cannot override failure. No public benchmark result or independent model-review PASS is claimed.

The negative-sampling training idea is informed by the official UVLTrack actor at commit 6ca34055c3447cd69b032eeb0f7cf6af6c9f3728; its code, floor coordinates and full architecture are not copied. Project-specific loss alignment is a hypothesis under test, not a pre-established language contribution.
