#!/usr/bin/env python3
import argparse
import json
import os


def _load_one(path):
    with open(path, "r") as f:
        data = json.load(f)
    if isinstance(data, list):
        if not data:
            raise ValueError("{} is empty".format(path))
        data = data[0]
    if not isinstance(data, dict):
        raise ValueError("{} does not contain a DepthTrack result dict".format(path))
    return data


def _fmt_delta(value):
    return "{:+.2f}".format(value)


def _per_sequence(result):
    return {
        item["sequence"]: item
        for item in result.get("per_sequence", [])
        if isinstance(item, dict) and "sequence" in item
    }


def main():
    parser = argparse.ArgumentParser(description="Compare two DepthTrack Pr/Re/F-score JSON result files.")
    parser.add_argument("reference", help="Reference/baseline JSON")
    parser.add_argument("candidate", help="Candidate JSON")
    parser.add_argument("--topk", type=int, default=12, help="Number of per-sequence deltas to print")
    parser.add_argument("--min-full-f1", type=float, default=63.6, help="Full-eval F-score target")
    parser.add_argument("--min-full-pr", type=float, default=64.1, help="Full-eval precision target")
    parser.add_argument("--min-full-re", type=float, default=63.1, help="Full-eval recall target")
    args = parser.parse_args()

    ref = _load_one(args.reference)
    cand = _load_one(args.candidate)

    print("reference: {} ({})".format(os.path.basename(args.reference), ref.get("run_id", "unknown")))
    print("candidate: {} ({})".format(os.path.basename(args.candidate), cand.get("run_id", "unknown")))
    print("")
    for key in ("Pr", "Re", "F-score"):
        r = float(ref.get(key, 0.0))
        c = float(cand.get(key, 0.0))
        print("{:<7} ref={:6.2f} cand={:6.2f} delta={}".format(key, r, c, _fmt_delta(c - r)))

    seq_avg_ref = ref.get("sequence_average", {}) or {}
    seq_avg_cand = cand.get("sequence_average", {}) or {}
    if seq_avg_ref or seq_avg_cand:
        print("seq-F  ref={:6.2f} cand={:6.2f} delta={}".format(
            float(seq_avg_ref.get("F-score", 0.0)),
            float(seq_avg_cand.get("F-score", 0.0)),
            _fmt_delta(float(seq_avg_cand.get("F-score", 0.0)) - float(seq_avg_ref.get("F-score", 0.0)))))

    ref_seq = _per_sequence(ref)
    cand_seq = _per_sequence(cand)
    common = sorted(set(ref_seq) & set(cand_seq))
    missing = sorted(set(ref_seq) - set(cand_seq))
    extra = sorted(set(cand_seq) - set(ref_seq))
    print("")
    print("common sequences: {}  missing in candidate: {}  extra: {}".format(
        len(common), len(missing), len(extra)))
    if missing or extra:
        print("note: aggregate Pr/Re/F-score above are not directly comparable because sequence sets differ; use per-sequence deltas.")
    if missing:
        print("missing:", ", ".join(missing[:20]) + (" ..." if len(missing) > 20 else ""))

    deltas = []
    for name in common:
        r = ref_seq[name]
        c = cand_seq[name]
        deltas.append({
            "sequence": name,
            "d_pr": float(c.get("Pr", 0.0)) - float(r.get("Pr", 0.0)),
            "d_re": float(c.get("Re", 0.0)) - float(r.get("Re", 0.0)),
            "d_f": float(c.get("F-score", 0.0)) - float(r.get("F-score", 0.0)),
            "ref_f": float(r.get("F-score", 0.0)),
            "cand_f": float(c.get("F-score", 0.0)),
        })

    if deltas:
        print("")
        print("largest F-score drops:")
        for item in sorted(deltas, key=lambda x: x["d_f"])[:args.topk]:
            print("- {sequence:<28} F {ref_f:6.2f}->{cand_f:6.2f} dF={d_f:+6.2f} dPr={d_pr:+6.2f} dRe={d_re:+6.2f}".format(**item))
        print("")
        print("largest F-score gains:")
        for item in sorted(deltas, key=lambda x: x["d_f"], reverse=True)[:args.topk]:
            print("- {sequence:<28} F {ref_f:6.2f}->{cand_f:6.2f} dF={d_f:+6.2f} dPr={d_pr:+6.2f} dRe={d_re:+6.2f}".format(**item))

    meets_target = (
        float(cand.get("Pr", 0.0)) > args.min_full_pr and
        float(cand.get("Re", 0.0)) > args.min_full_re and
        float(cand.get("F-score", 0.0)) > args.min_full_f1
    )
    print("")
    print("target gate: {}".format("PASS" if meets_target else "FAIL"))


if __name__ == "__main__":
    main()
