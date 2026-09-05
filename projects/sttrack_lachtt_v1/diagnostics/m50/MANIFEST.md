# M50 artifacts

- RESULT_REPORT.md: completed full Train recursion; integrity PASS, performance FAIL.
- Frozen spec and runtime unchanged; recursive.exit=0.
- analysis.log / analysis.exit=1 preserve the first reporting error.
- finalize_m50.py reads complete SHA-bound native baseline traces; analysis_r2.exit=0.
- recursive_result.json, per_sequence.csv, recursive_receipt.json are the verified outputs.
- native_full127 remains a separate native baseline run, not M50 advancement.
