# Sparse R0 training-state description

This is a retrospective CPU-only analysis of the six completed R0 training
passes. It is not evaluation of one frozen final checkpoint.

- `diagnostic/report/summary.json` records sampling, per-pass counts, source/GT hashes and interpretation limits.
- `diagnostic/report/samples.csv` contains all3996 sampled states with aligned GT and geometric measurements.
- `diagnostic/run_receipt.json` records the successful CPU run and geometry checks.
- `diagnostic/code/` preserves the executed analyzer.
- `original/` preserves the locked training specification, gate, completion result, six original traces/results and relevant source.

Raw bytes are preserved in Git. The source archive SHA256 is
`4032cd76c191f714c98a7589043592e91a1a64087643e7f05bbfc416c6d4ce9b`.
The19 archived inputs represented in the summary were rechecked after download;
the152 complete GT files remain in the dataset and are represented by their
verified hashes and sampled GT rows. See master-document section5.561 for the
results table and the required distinction between sparse training states,
full frozen-model trajectories, and benchmark outcomes.
