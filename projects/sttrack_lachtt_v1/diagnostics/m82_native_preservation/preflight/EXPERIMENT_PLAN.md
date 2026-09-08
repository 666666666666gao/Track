# M82: reliable same-state native spatial preservation

Status: frozen after causal checks; no formal training started at freeze.

M81 completed negatively (§5.168). M82 returns to matched M78 t0/full-forward Raw-competition training. The only learning change is weight-1 spatial KL on native-correct training samples. Seed2027 only, 130 fit sequences, 186694 tracking calls, same 32-frame valid-loss averaging and fixed final checkpoints. Train Category and Empty in parallel on the two existing GPUs; reuse sealed M78 counterparts for the loss increment comparison. Both Qwen models and all published final heads are retained.

Teacher uses the already-computed native raw response and its Hann-selected box on the current student's causal state. Decode before student state changes; never commit teacher state. Only after the current prediction, GT enables the loss for valid, centre-inside samples with native IoU>=0.5. Invalid, crop-outside and native-wrong samples receive no KL. The original tracking/Raw competition losses remain unchanged. Normalize positive raw sigmoid response over 256 positions; sum KL(teacher||student), batch mean, weight1. No extra ViT, inference head, template rule, dropout, caption or multi-start changes.

This does not guarantee independent native trajectory recovery or absolute confidence preservation. It is not a new quality gate or proven language innovation.

All four dev22 families are sealed before analysis: Category, independently trained Empty, Category-head Empty, Category-head Swapped. Ten existing native/paired-Empty gates, eight content gates, four matched-M78 Category increment gates are fixed prospectively. Empty-vs-M78 Empty increment is descriptive. Report pooled/macro IoU, low frames, H10, success damage and leave-one-sequence-out content deltas. No automatic external evaluation. The same final bundle must eventually satisfy all three official datasets; no prior-model metric splicing.

Preflight: exact 101-frame zero-residual native bbox/query/template parity for both arms, detached teacher; independent numerical KL reference; invalid/outside/wrong teacher masks; disabled-KL exact loss and supervised output-gradient equivalence; selected parameter gradients compared at rtol1e-5/atol1e-7 with repeated raw-backward diagnostics; real 96-frame/3-update causal smoke per arm; frozen base/buffer hash and source checks. Smoke weights discarded. Environment warm-reused without dependency changes.

Independent gpt-6-astra/max review remains unavailable after previously recorded quota rejection until Sep12; this run makes no new review call and does not substitute another model. Deterministic checks are not independent model review.

References: PromptSRC (ICCV2023), official trainer commit bb95c77b634d63488f2cad81ff4a72d53bdd06d5; CoPrompt official trainer commit a2a6c12e3622cfe4b3a127f4ba0c4f3eb841418c. Borrow functionality-preservation principle; neither establishes this tracking recipe's effectiveness.

Preflight attempts1/2 stopped in the checker (not formal training): bitwise parameter gradients differed at ~1e-8, also observed when repeating the same raw loss; crop-outside samples have no size/offset regression gradients. Checker now compares only supervised outputs exactly and reports parameter-level roundoff. Training implementation and coefficient unchanged. Evidence logs retained.
