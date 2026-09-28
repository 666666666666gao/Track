# M100 completed fixed-state score diagnosis

Actual GPU smoke and full readout both exit0. Smoke:64 fit events,10.886s;
full:2544fit+495development=3039events,12.996s. No optimizer/checkpoint,
tracker state, new labels, runtime policy or official benchmark was used.
The fixed final weight is M99's53d60b477cc9d3c81a923e494686e658218a4c5b8facdd2b77ac8e1766ac76f2.
All495development rows and both selection summaries exactly reproduce M99.

| split | native correct | selection correct / rescues / breaks | quality argmax correct / rescues / breaks | selection meanIoU | quality meanIoU | quality candidate MAE |
|---|---:|---:|---:|---:|---:|---:|
| fit | 1575/2544 | 1632 / 61 / 4 | 1644 / 80 / 11 | 0.611696918 | 0.608354957 | 0.107881677 |
| development | 268/495 | 275 / 10 / 3 | 275 / 12 / 5 | 0.530180954 | 0.526050107 | 0.126265304 |

Quality argmax is a readout, not an independently deployed model or proof of
physical identity. It shares the visual encoder with selection. Development
healthy quality breaks4 versus selection2; replacing selection with quality
does not pass the protection check.

All three selection development breaks still have a quality argmax with
IoU0. At egg_indoor@46 quality chooses candidate7 rather than selection1,
but both miss. The other two choose the same wrong candidate. This rules out
the proposed direct quality-head substitution on these saved states; it does
not rule out a better trained quality mechanism.

| development break | native gap to selected | residual advantage | final selection gap | native IoU | selected IoU | quality-choice IoU |
|---|---:|---:|---:|---:|---:|---:|
| egg_indoor@10 | 1.927375 | 1.951734 | 0.024359 | 0.799761 | 0.000000 | 0.000000 |
| egg_indoor@46 | 0.193628 | 1.036610 | 0.842982 | 0.698304 | 0.000000 | 0.000000 |
| egg_indoor@2439 | 1.235834 | 2.179558 | 0.943723 | 0.863486 | 0.000000 | 0.000000 |

The final rank reversal is numerically observed: learned residual advantage
exceeds the native gap. This is the arithmetic of selection, not a unique
causal explanation for encoder errors. Native scores are log(clamped Hann
response), not log odds; the mismatch in interpretation alone does not prove
a loss bug. No score transformation, threshold or checkpoint search occurred.

On fit, selection has four breaks (lock01@392, pigeon05@37, cube06@749,
earphone02@294); quality argmax is also below.5 in all four. Two are IoU0,
two are near-boundary/localization mistakes. Their GT-IoU labels do not
establish that all four candidates are different physical objects.

M101 prepares one predeclared0/1 reliable-native-preservation pair in the
new B decoder. This reuses M82/M89's preservation principle; it is not a new
semantic contribution. Its weight0 must reproduce the preserved M99 final,
and either arm must retain all development gains and harms. No semantic or
recursive promotion has occurred. The original M99 healthy gate remains FAIL.
