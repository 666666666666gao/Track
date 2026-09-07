# M76 exact-state content and response diagnostic

Fixed seed2027 M73 Category final checkpoint. Four selected Train development harm windows, 6925 exact public-prefix tracking calls and 429 probe frames. The adapter receives the same actual six inputs, with only text tokens changed for Empty and Swapped. Unadapted Center Head probes use the same Category-state fused features. None of the probes commits bbox, queries, templates or semantic state.

All public boxes/scores match the original Category trajectories exactly. CPU artifact audit independently recomputes prefix geometry, overlaps, rankings and native Hann coefficients. This is artifact/scalar verification, not an independent learned-model review.

At the book06 and cup08 H10 onsets, none of the four heads has a dense decoded box with IoU >= 0.5. At cup10 and egg, the correct Category candidate ranks second after Hann, while its raw score ranks first. Across the selected windows, choosing raw Category maxima would recover 33/228 low-overlap frames but severely break 7/12 previously correct frames. These post-result window statistics do not establish a deployable policy or an independent recursive recovery.

No new training, seeds, checkpoint or public benchmark result. M73 frozen promotion failure is unchanged. Full raw response/dense-box replay files remain on the server; their hashes are recorded in receipt.json and completed_evidence_audit.json. Compact per-probe metrics and peak timelines are included here. See project master section 5.143.
