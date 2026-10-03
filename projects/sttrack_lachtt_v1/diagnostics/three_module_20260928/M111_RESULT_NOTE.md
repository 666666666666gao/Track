# M111 completed: explicit initial-category evidence does not improve selection

Both RTX3090 arms completed the separate3-update sanity and fixed12-epoch,
480-update final training. Preparation, sanity, training, controller and driver
all exited0. Actual total driver runtime77.178341seconds; final-optimizer loops
were32.19/29.42seconds. Both optimize95683 semantic parameters, preserve all
other parameters/buffers, and retain exactly identical Empty scores/quality on
all3039 cached states. No recursive training or official evaluation was started.

Zero-weight final SHA256 `36f3cdf300522eeb68252f538d04c4b8e02eff6a69a45c09102c3bebf82da4cc`
equals the completed M110 reviewed-text final. This demonstrates this particular
fixed-state control reproduced that predecessor, not general determinism of
CUDA or recursive training. Weight1 final SHA256 is
`427dedb96497411ad0ebb712b24d0fb02d2a00c9f07cf6579a7e0fedcb21b1fe`.

The new supervision is agreement with the authorized model's initial review,
not independent semantic truth. Its98 fit/19 development original-category
queries are separate from the corrected descriptions used for candidate selection.

| Final | Fit category correct /98 | Fit CE | Development correct /19 | Development CE | Dev support/conflict/uncertain recalls |
|---|---:|---:|---:|---:|---|
| Weight0 |39|1.105401|9|1.017831|.600/0/0|
| Weight1 |98|.000982|15|2.011453|1.000/0/0|
| Constant majority-class reference |80|—|15|—|1.000/0/0|

Weight1 predicts supported for every development query. Its15/19 agreement is
exactly the majority-class reference; all3 conflict and1 uncertain examples are
missed. The same-query/different-label subset is21/21 on fit and2/3 on development;
2/3 is also achieved by always predicting supported there. Perfect fitting
agreement is not transferable verification. The higher development CE indicates
the errors are costly in probability space; no unique cause such as learning
rate, feature capacity, wrong labels or visual neglect has been established.

| Final/content | Dev correct IoU>=.5 /495 | Mean IoU | Rescues / harms versus native |
|---|---:|---:|---|
| Native cached reference |268|.524770723|—|
| Frozen M101 visual parent / Empty, both finals |272|.530707698|5/1|
| Weight0 generic |272|.530707698|5/1|
| Weight0 reviewed description |272|.530402968|5/1|
| Weight1 generic |272|.530707698|5/1|
| Weight1 reviewed description |272|.530707698|5/1|

Relative to its own Empty, weight0 changes6 selections:0 IoU improvements and1
worsening, mean delta-.000304730. Weight1 changes3 selections but all3 have the
same IoU:0 improvements,0 worsenings, mean delta0. Both satisfy the old four
capacity checks, keep264/264 healthy cases, and have4/127 transition successes.
Those inherited visual checks do not demonstrate semantic benefit. Fit candidate
correct counts are1605/2544 for weight0 and1603/2544 for weight1.

CPU recount independently verified all2544 fitting and495x3 development rows per
arm, fit/dev separation, all subgroup aggregates, own-Empty pairing, confusion
arithmetic, input hashes and checkpoint hashes. Category CE itself was not
recomputed from logits and no checkpoint forward was replayed in that recount.
The32 downloaded artifacts/4171459bytes were checked against remote bytes;
binary checkpoints and category bank remain private. Source review is a fresh
gpt-6-astra/max same-family provisional PASS after the actual FP16-cache input
issue was fixed; it is separate from actual GPU completion and scientific success.

Do not promote this final into recursive or public evaluation, repeat the same
run, or tune the coefficient/checkpoint on these development results. Next work
should bind explicit semantic labels to the exact current-candidate query and
observation, distinguish supported/conflicting/unknown with appropriate evidence,
and separate initial identity binding from current-region verification. An
initial-only auxiliary classifier cannot by itself establish candidate identity
or visibility. Use training-only states and preserve the visual/Empty control;
avoid inheriting the old24 candidate labels onto corrected phrases or labeling
every low-IoU box as a different physical instance. That next implementation has
not yet been launched. The joint three-dataset goal remains unmet.
