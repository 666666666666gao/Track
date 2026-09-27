# M90 Train-state collection review

The independent static review found one blocking plan/code mismatch: the existing
candidate helper returns 10 NMS peaks, while the original plan said it cached
features for all 256 search cells. At 3502 events and 768 channels, 256 RGB-D
4×4 fp16 RoIs would take approximately 41 GiB, exceeding the 4.59 GiB free
remote disk space observed before collection.

The collector and plan now distinguish all 256 decoded boxes and dense
score/response/size/offset maps from RGB-D RoI features for only the 10 NMS
peaks. Each saved peak has a cell index. Collection verifies that its box
matches the corresponding dense box, as well as native public bbox/score
parity on every replayed frame. Train-only GT remains in a separate labels
file that the feature collector does not open. The reviewer found no further
static blocker after this correction.

GPU smoke must still verify actual parity, tensor shapes, and storage size
before full collection. This review is not a performance result or proof of
language use.

The later two-GPU collection queue also received independent static review.
Its first version would exit on a failed shard0 wait without waiting for
shard1, and would leave no terminal receipt on failure. The queue now records
both shard exit codes before deciding whether to analyze, and an EXIT trap
records the script's real terminal status. The verifier checks smoke parity,
artifact shapes, and free space based on measured smoke bytes before the
two-shard launch. The reviewer found no remaining concrete static blocker.
GPU smoke and full collection remain unverified until their own receipts exist.
