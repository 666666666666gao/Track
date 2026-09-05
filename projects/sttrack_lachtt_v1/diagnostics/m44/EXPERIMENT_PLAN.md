# M44 execution plan

| Stage | Required evidence | Expected work |
|---|---|---|
| Contracts | Nontrivial set-permutation check; exact geometry ablation; real optimizer descent | Small synthetic test |
| Native collection smoke |120 frames, six pairs; default bbox/score and adjacent-frame alignment | Two fit sequences |
| Collection | All85 sequences and frozen pair requests, raw FP16 observations, complete hashes | Two3090 GPUs, about60 minutes |
| Fixed training | Both final heads,20 epochs, separate source/weight binding and strict reload | About5–15 minutes after collection |
| Full recursive development | Same22 sequences, independent complete state, sealed trajectories before GT analysis | About35–45 minutes, two GPUs |
| Promotion | Frozen geometry primary passes all performance conditions | Then low22; no public automatic launch now |

The original model/checkpoint are unchanged. New head weights are trained only
on DepthTrack Train. Static and mechanism results do not replace the full
recursive performance question. Disk estimate is bounded by at most3230 pairs
at roughly1.08MB each, within the measured6.8GB free space; no old evidence is
deleted. Collection and training receipts will replace launch status in the
tracker, preserving its launch version under history/.
