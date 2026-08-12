#!/usr/bin/env python3
import argparse
import json
import os
import shutil
from pathlib import Path


def _load_metric(path):
    with open(path, "r") as f:
        data = json.load(f)
    if isinstance(data, list):
        if not data:
            raise ValueError("empty result list: {}".format(path))
        data = data[0]
    if not isinstance(data, dict):
        raise ValueError("unsupported result json format: {}".format(path))
    return data


def _score(metric, mode):
    if mode == "f":
        return float(metric.get("F-score", 0.0))
    if mode == "all":
        return (
            float(metric.get("Pr", 0.0)),
            float(metric.get("Re", 0.0)),
            float(metric.get("F-score", 0.0)),
        )
    raise ValueError("unknown mode: {}".format(mode))


def _tag(metric, suffix):
    tag = "full50_pr{:.2f}_re{:.2f}_f{:.2f}".format(
        float(metric.get("Pr", 0.0)),
        float(metric.get("Re", 0.0)),
        float(metric.get("F-score", 0.0)),
    )
    tag = tag.replace(".", "p")
    if suffix:
        tag = "{}_{}".format(tag, suffix)
    return tag


def main():
    parser = argparse.ArgumentParser(
        description="Copy a DepthTrack checkpoint into KEEP_BEST when it beats a reference result.")
    parser.add_argument("--candidate_json", required=True)
    parser.add_argument("--reference_json", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--keep_root", default="output/depthtrack_roberta/checkpoints/KEEP_BEST")
    parser.add_argument("--mode", choices=["f", "all"], default="f",
                        help="f requires higher F-score; all requires lexicographically higher Pr/Re/F.")
    parser.add_argument("--suffix", default="")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    cand = _load_metric(args.candidate_json)
    ref = _load_metric(args.reference_json)
    cand_score = _score(cand, args.mode)
    ref_score = _score(ref, args.mode)

    print("candidate: Pr={:.2f} Re={:.2f} F={:.2f}".format(
        float(cand.get("Pr", 0.0)), float(cand.get("Re", 0.0)), float(cand.get("F-score", 0.0))))
    print("reference: Pr={:.2f} Re={:.2f} F={:.2f}".format(
        float(ref.get("Pr", 0.0)), float(ref.get("Re", 0.0)), float(ref.get("F-score", 0.0))))

    if cand_score <= ref_score:
        print("not protected: candidate does not beat reference in {} mode".format(args.mode))
        return 1

    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(str(checkpoint))

    keep_dir = Path(args.keep_root) / _tag(cand, args.suffix)
    dst = keep_dir / checkpoint.name
    print("protecting checkpoint: {}".format(dst))
    if args.dry_run:
        return 0

    keep_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(str(checkpoint), str(dst))
    with open(keep_dir / "metrics.json", "w") as f:
        json.dump({
            "candidate_json": os.path.abspath(args.candidate_json),
            "reference_json": os.path.abspath(args.reference_json),
            "source_checkpoint": os.path.abspath(str(checkpoint)),
            "metric": cand,
            "mode": args.mode,
        }, f, indent=2)
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
