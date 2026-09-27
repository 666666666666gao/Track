# M92 blinded candidate review packet code review

Independent same-family reviewer: `/root/m91_selector_review` (2026-09-28).
Acceptance: provisional PASS, no blocking issues. The reviewer did not run the
packet generator or modify files.

The reviewer checked the former-fit130-only selection, `xywh` boxes,
`frame+1` image path, one event per sequence, 24 selected cases, seed2027
balanced A/B placement, and separation of private GT/candidate identity from
the A/B images. The remote root/spec bind the already collected Train cache.
Actual source-image presence and the completed output were checked by the main
executor; the reviewer could not independently access the remote data.

The packet is deliberately selected for clear candidate disagreement and is
not a random sample or a semantic truth set. The local reviewer page also
shows the immutable first-frame target alongside the current candidates;
human conclusions remain blank.
