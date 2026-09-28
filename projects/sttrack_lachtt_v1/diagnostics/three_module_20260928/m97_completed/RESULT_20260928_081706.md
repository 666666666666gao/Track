# M97 actual-input A+B interface: PASS, wiring only

Fresh-context gpt-6-astra/max source review passed after correcting the native
resize precision and sampling the actual crop padding mask. Review independence
is same-family and acceptance provisional. The review's CPU synthetic checks
were followed by actual GPU inputs; neither is a semantic performance result.

One-case smoke on GPU0 exited0 in9.75s. Two four-case collection shards on
RTX3090 GPUs0/1 exited0 in11.95/11.56s. Their eight prescribed former-fit Train
sequences were cube04, bag04, skateboard02, cat02, bottle02, bottle01,
mushroom02 and book04 (full names in the receipts), all at frame10. The80
native prefix calls had maximum bbox and score errors0. All event proposal
boxes and half-precision local RoIs exactly matched M90; legal t0 search RoIs
exactly matched M95. The auxiliary query was not committed. Collection loaded
no subsequent GT or text; twelve surrounding samples and actual padding support
were added to the existing sixteen local samples. Raw nonzero depth fraction
is an input descriptor, not a calibrated reliability probability.

The actual-input probe on GPU1 exited0 in3.98s. This isolated A+B prototype has
241,994 trainable parameters. On eight valid Train-GT IoU targets, two throwaway
AdamW steps checked gradient wiring; losses were2.915009/2.879848. No trained
checkpoint was saved. Finite nonzero second-step gradient norms were:

| Path | Gradient norm |
|---|---:|
| Visual reference read | 0.08646554 |
| Text projection | 0.0005621113 |
| Visual to phrase | 0.00001995577 |
| Phrase to visual | 0.0003155169 |
| Phrase evidence | 0.0001059467 |
| Candidate relation | 0.20661889 |

Zero initialization exactly preserved the native candidate0 choice. After
updates, empty semantic and phrase increments were exactly0 and Empty selection
exactly equaled the visual branch. Localization quality was independent of text
both by equal-output checks and unused text gradients. Candidate permutation
maximum error was0 for selection and1.1920929e-7 for quality. Final selected
box, score, quality and feature all used the same index. Outputs were finite.

These checks establish the small-panel interface, not learned semantic support,
physical identity, candidate-selection gains, recursive safety or official
tracking metrics. The existing category bank remains unverified and active
attribute slots are empty. Phrase evidence received indirect localization
gradients without semantic-state supervision or calibration. The observation
head remained untrained. C memory/observation control and local box refinement
were not deployed; geometry was frozen for this first stage.

Raw JSON/logs are preserved beside this report. `interface_sanity.json` SHA256
is `9884018c377c74678b3a2bd506f258394eb5dbb550e5fd8de5387ec20a7de7ed`.
Panel tensors remain at
`/root/autodl-tmp/sttrack_m97_ab_interface_20260928/`, not on GitHub.
No public dataset, tracker-state submission or official checkpoint was used.
The human-reviewed phrase/candidate labels and content-versus-visual development
gate remain separate requirements before semantic training and recursion.

All jobs are finished. M67/M82/M89 formal results and the unmet nine-metric
goal are unchanged.
