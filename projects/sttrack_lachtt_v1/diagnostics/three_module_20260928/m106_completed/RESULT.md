# M106 selected-geometry preservation: completed negative comparison

Both two-update sanities and both fixed12-epoch/384-update arms actually completed on GPU0/GPU1, seed2027. The driver elapsed38.521107seconds. This is fixed DepthTrack Train-cache visual geometry learning, not recursive tracking, semantic contribution, Full152 sequence retraining or official benchmark results.

The M104 architecture/59540parameters, M101 parent, Empty slots, 2033eligible fit events, initialization, order, AdamW3e-4, batch64, GT-best localization objective and final selection are unchanged. Weight1 adds the batch mean of 1[parent selected GT IoU>=.5]*clamp_min(parent IoU-refined selected IoU,0). No GT enters forward, candidate selection or an inference gate. Training GT establishes localization only.

| split | readout | selected correct | mean IoU | parent rescues/breaks | top10 oracle |
|---|---|---:|---:|---:|---:|
| fit | frozen parent | 1602 | 0.608313788776 | 0/0 | 1814 |
| fit | weight0 | 1652 | 0.616710538305 | 51/1 | 1866 |
| fit | weight1 | 1641 | 0.616584581926 | 40/1 | 1856 |
| development | frozen parent | 272 | 0.530707697987 | 0/0 | 305 |
| development | weight0 | 286 | 0.535297504355 | 15/1 | 323 |
| development | weight1 | 282 | 0.535005414295 | 11/1 | 317 |

Weight0 exactly reproduces M104 checkpoint bytes/all tensors/all3039 ordered rows and every original summary. Thus this pair does not have the unresolved control reproduction difference seen in the earlier recursive M89 experiment. It does not establish determinism of other environments or recursive runs.

Weight1 loses11fit and4development correct states relative to weight0, gains0in either split. All15 lost-correct states are tagged intermediate. Healthy break counts remain1in both splits. The same parent-correct fit pine01_indoor@1276 improves from weight0 .47923150658607483 to weight1 .49121901392936707 but remains below .5. Development mobilephone02_indoor@65 improves from .44826263189315796 to .4675859808921814 but remains below .5; parent was .5105010271072388.

Both final arms pass3/4original capacity checks and fail zero healthy new breaks. Transition selected correctness stays at parent22fit/4development. No weight/epoch/seed scan, favorable component selection, changed gate, C/recursive/public promotion or formal nine-metric update is authorized by these results.

Every weight1 epoch consumes1591reliable selected fit calls; preservation mean is .0018825707738869824 at epoch1 and .0035323620249982923 at epoch12. Loss is active but not eliminated. These values do not uniquely diagnose gradient conflict or prove a particular weight adjustment would succeed.

Read-only checkpoint replay writes all91170candidate boxes (3039states*10candidates*parent/weight0/weight1), reruns both saved finals over allstates, and verifies frozen states. Independent local CPU float32 GT arithmetic checks91170unique candidate IoUs (182340 repeated arm-specific comparisons), all reported marginal summaries and all original gates exactly. All25 downloaded runtime files match remote SHA/bytes. Dataset-GT cache source is the prior hash-bound training_labels.json; no independent human identity truth is supplied.

Final weight0 SHA256: 546bd218a14906beb28033717bb53cc17721e9bb6b3fb53f36a4bac15dff9f6e
Final weight1 SHA256: 57082505db00dd81467e5060a2145da6e365e3807f4580790d0a7c0cc9ccca25

Next: inspect fitting-state candidate support and the GT-best-versus-actual-selected training mismatch before defining a new training task. Keep the original negative result. Do not respond by scanning the preservation coefficient or fitting a phone-specific rule. User-reviewed semantic labels remain pending; independent visual work is possible.

Fresh completed integrity review is pending at the time this result file is written. No independent verdict is asserted here.
