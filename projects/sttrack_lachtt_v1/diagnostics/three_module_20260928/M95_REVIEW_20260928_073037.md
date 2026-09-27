# M95 predeployment code and scientific-integrity review

- verdict: PASS
- reviewer: fresh Codex reviewer, gpt-6-astra / max
- review_independence: same-family
- acceptance_status: provisional
- scope: source review only; no deployment, training, or GPU inference performed by this reviewer

No concrete blocking defects found. No code changes requested.

Reviewed `M95_INITIAL_ORIGIN_PLAN.md`, `collect_initial_feature_origins.py`, `audit_initial_feature_origins.py`, the existing M90 collector and M91 audit, the IoU helper, and the local overlay's native tracker, model, configuration, and spatial observation functions. Also inspected the saved native backbone and TSG interfaces in the M84 source snapshot. Runtime equivalence to the deployed repository remains the planned smoke check.

## Findings

1. **The two references implement the stated t0 origin.** Collector lines 51-66 initialize from the legal first RGB-D frame and initialization box, then use the same frame for the auxiliary search crop. `template_roi(..., 0, bbox)` reads the first template's post-TSG tokens; `search_rois(..., bbox, resize)` reads the initial search tokens at the legal box. Both use the existing 4 by 4 bilinear helper. The crop origin and resize conventions agree with the existing `sample_target` implementation. Both references are t0-only in pixel provenance, while still being jointly contextualized representations from the t0 template/search forward.

2. **Auxiliary results are not committed to tracking state.** Collector lines 57 and 69 check frame 0, unchanged box values, and absent historical query before and after the auxiliary call. The call passes `track_query_before=None` and never assigns its returned query. Native initialization resets semantic context to `None`; model forward's omitted semantic argument defaults to `None`. Native construction calls `eval()`, and the auxiliary call is under `torch.no_grad()`. The inspected backbone derives embeddings from the input template tensors, without overwriting those image tensors. The native first real frame is then checked against the saved native bbox and score at tolerances 1e-4 pixels and 1e-6 respectively (collector lines 70-77).

3. **Collection and label evaluation are separated.** The collector reads inference inputs, the source specification, first RGB-D frame, and second RGB-D frame for native verification. It does not open subsequent ground-truth files, training labels, semantic labels, or candidate answers. `prepare_train_states.py` shows that `expected_rows` contains saved native predictions and scores, not GT. Using those predictions here is a trajectory-equivalence assertion, not the evaluation metric. The audit evaluates selected boxes against `training_labels.json` via the existing IoU helper.

4. **The comparison holds candidates, numerical representation, and scoring rule fixed.** Audit lines 46-79 reuse existing M90 candidate RoIs and boxes. All three references and the candidates are converted from the stored half tensors to float, mean-pooled over the 16 RoI samples, scored by per-modality cosine averaged across RGB and depth, and selected by the same argmax as M91. The first-use baseline summary is required to equal the saved M91 summary for both splits. No model fitting, trajectory update, semantic input, expanded candidate search, memory write, or official metric is introduced.

5. **Coverage and reporting follow the planned diagnostic.** Both complete non-smoke shards are loaded; the audit requires 152 unique reference sequences, matching split identifiers, and 10,506 variant-event rows (three times the 3,502 M90 labels). It reports native and Top-10 coverage, visual choices, rescues, breaks, mean IoU, and all supplied strata for fit and development. The existing prepared data establish the fit130/development22 split. The collector's records retain sequence and split identity, and reports retain the native-equivalence errors.

## Required execution gate and interpretation

Run the already planned one-sequence smoke before both full shards. A source review does not establish GPU kernel execution, the actual remote checkpoint/configuration, or first-frame numerical agreement. The collector's assertions must pass in that smoke and all full-shard cases before interpreting the cosine audit.

The result can measure how reference origin changes fixed-state candidate choices. It cannot establish a language effect, causal explanation of M91's weak result, recursive tracking improvement, or an official nine-metric improvement. The plan already states these limits and correctly conditions any later visual-control training on measured benefit.

No new defensive framework, fingerprint scheme, fallback, or unrelated refactor is warranted by the reviewed code. No private candidate-answer manifests or memory files were read. No external model reviewer was used; this PASS is same-family and provisional.
