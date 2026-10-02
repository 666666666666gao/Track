# M109 fitting-state gradient diagnosis

Both GPU diagnostics completed with zero exits. Driver runtime16.775 seconds.
There was no optimizer, training update, new checkpoint, development model
evaluation or public benchmark evaluation. Parameters and buffers were verified
unchanged at each snapshot. The result files retain the raw per-batch quantities.

Each arm measured40 fixed batches /2544 fitting states at initialization and at
its M108 final. Eleven provisional A/B pairs appeared in9 batches. Ten selected
preferences have greater candidate IoU; one has lower IoU. This is a localization
ordering disagreement, not proof of a wrong physical-identity annotation.
Private event identifiers and judgments remain outside the public repository.

| Text arm / snapshot | Mean weak gradient norm divided by localization+preservation norm in paired batches | Mean cosine with localization+preservation | Opposing paired batches |
|---|---:|---:|---:|
| Generic / initial |2.871630|−0.000996|4/9|
| Generic / final |2.190598|−0.106145|7/9|
| Reviewed / initial |3.801637|−0.007277|4/9|
| Reviewed / final |0.915496|0.187322|1/9|

The weak term is large at initialization, but reviewed-text endpoint gradients
are usually aligned. These measurements do not establish persistent gradient
conflict, reconstruct historical updates, or explain generalization uniquely.
The generic endpoint has more opposing batches despite its better M108 held-out
result, which further rules out a simple one-number explanation.

The next controlled test is M110: retain captions, initialization and budget,
exclude the provisional A/B term from optimization, and run both text arms.
This tests the contribution of the auxiliary objective rather than correcting
an identity label based solely on IoU. This diagnostic performed no new official three-dataset evaluation.
