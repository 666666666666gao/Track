# Exact native tracker source used by M52

`sttrack.py` is a byte-for-byte source snapshot read from
`/root/autodl-tmp/rgbd_baselines/STTrack_lachtt_v1/lib/test/tracker/sttrack.py`
on 2026-09-05. Its SHA-256 is
`d67d551a612b80cee5b19a00f6fecd5d0f7ed0c907e800f452873afd684cc58f`
and its size is 10,216 bytes. This matches the source binding in
`diagnostics/m44/spec.json` used by the M52 collection and training checks.

This file supplies the native template/query/state implementation for source
review. It is not a new algorithm or a change to the running tracker. The
project's `overlay/` remains an overlay on the STTrack runtime checkout, not a
standalone upstream source distribution.

The first M52 advisory review was completed before this snapshot was supplied;
its original warning is preserved in `../EXPERIMENT_AUDIT.md`.
