# M43 execution

- Verify M42 terminal integrity, its failed superiority gate and the exact tied
  snapshot choices of its two trained heads. Freeze pooled as primary.
- Launch pooled on GPU0 and spatial on GPU1, each with independent full state
  over all33,130 development frames. No new training or threshold search.
- Seal complete trajectories and validate source/checkpoint hashes.
- Read GT only in posthoc analysis; compare both arms with cached default and
  report pooled's preregistered primary gate.
- On primary pass, prepare the frozen low22 comparison; otherwise retain the
  actual recursive failure evidence and launch no public benchmark.

Run root: `/root/autodl-tmp/sttrack_m43_pooled_recursive_v1_20260905`.
Controller: `screen sttrack_m43_recursive_20260905`, started2026-09-05 at11:32
server time. Estimate runtime from its first completed sequence logs; inference
is larger than the static head fitting and requires sustained GPU work.
