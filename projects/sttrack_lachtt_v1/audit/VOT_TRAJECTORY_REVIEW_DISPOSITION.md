# Sealed ROI coverage review disposition

The fresh read-only gpt-6-astra/max review in `VOT_TRAJECTORY_REVIEW.md`
covered the original two-file implementation at commit6778495. It did not
run tests, inspect the installed VOT interfaces or read archived results.
Its attribution remains same-family/provisional. Its findings are retained;
this executor disposition is not a new reviewer PASS.

1. Assertion stripping is a real limitation of using this tool with Python
   optimization. The actual supported command is `python -B`, without `-O`
   or `-OO`; its environment and `__debug__` must be recorded with the result.
   Optimized execution is outside the documented scope. No general compatibility
   or fallback path was added for an invocation not used by this experiment.
2. The supplied execution digest is an external trust input, pinned to
   `51a3041dadbefb842622ccebd9253f79ea2bd507bc80796b0ac3b1f6244b06b1`
   for this archived run. Checking the listed current files is not proof of
   historic ledger completeness or checkpoint/source/plan provenance.
   Both full-binding and independent-completion flags remain false.
3. Direction-dependent coverage verifies expected decoded trajectory and
   auxiliary-stream lengths. It does not prove image processing order,
   prediction accuracy or neural generation history. No such claim is made.
4. The manifest deliberately rejects duplicate sequence/index filenames,
   and actual official anchors must match that partition exactly. The review
   did not verify the external anchor contract; a completed real-workspace
   run, rather than a speculative fallback, is required for this archive.

The real invocation additionally found two issues outside the static review:
the README named a nonexistent analysis file, and VOT0.7.1
`Trajectory.properties()` returns a tuple of names despite its dict return
annotation. The analysis path was corrected and auxiliary stream lengths
are now read from their actual files. A new test uses that tuple interface,
preserves blank first-frame values and rejects a truncated time stream.
All four local CPU tests passed after the fix. Failed invocation receipts
remain in their original remote attempt directories.
