# M101 completed new-B native-preservation control

Two RTX3090 GPUs ran the predeclared0/1 pair: actual sanity and full child
exits0, outer driver exit0. Start11:35:57CST, complete11:36:45CST; paired driver
48.757s, all input loading/sanity/training/evaluation/CPU reproduction included.
Both fixed12epochs/480updates/seed2027/final only,241994total/142086optimized.
No semantic label, tracker state, public benchmark or new memory was used.

The weight0 final is exactly identical to old M99 in every state tensor,
all495development rows, both split summaries, and saved weight SHA. This
actual no-feedback fixed-state reproduction addresses the current control
implementation; it does not prove all recursive training is deterministic.

| variant | development correct/495 | meanIoU | rescues | breaks | healthy breaks | predeclared checks |
|---|---:|---:|---:|---:|---:|---:|
| native | 268 | 0.524770723 | - | - | 0 | reference |
| weight0 | 275 | 0.530180954 | 10 | 3 | 2 | 3/4 |
| weight1 | 272 | 0.530707698 | 5 | 1 | 0 | 4/4 |

Preservation repairs healthy protection at the cost of fewer rescues and
three fewer total correct choices than weight0; meanIoU is slightly higher.
Neither learned variant dominates the other. All healthy264 events in
weight1 select candidate0 and exactly retain their native meanIoU.
The one remaining break is egg_indoor@46, transition,0.698304->0.
Its transition group still has one rescue and one break, so equal correct
count does not mean every transition is safe. All gains/harms and22sequence
rows are retained; no egg-specific rule, weight/seed/best-epoch search occurred.

Fit weight1 selects1602 versus native1575 and weight0's1632; rescues28/
breaks1 versus weight0's61/4. Each epoch contains2165 eligible preservation
pairs. Last preservation mean loss.007032407 and total mean loss1.038945135;
optimization23.862s excludes loading/evaluation. Sanity reserved832569344B
on each GPU, under1GiB. This is a short cached-state experiment, not Full152
recursive training; no hour-long run remains active.

The four predeclared visual-capacity checks now all pass for weight1.
This supports preparing the reviewed semantic A+B experiment; it is not
semantic content evidence, automatic runtime promotion or official
three-dataset acceptance. Physical competitor/phrase/visibility evidence
remains a separate supervision requirement. Keep independent human review
distinct from assistant screening and unreviewed automatic strings.
