#!/usr/bin/env python3
import argparse
import os
import sys
from collections import defaultdict

import torch


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def _state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("net", "state_dict", "model"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
    return checkpoint


def _group_name(name):
    for prefix in ("box_head", "mplt_fuse_search", "language", "backbone"):
        if name.startswith(prefix + ".") or prefix in name:
            return prefix
    return "other"


def main():
    parser = argparse.ArgumentParser(description="Compare BatchNorm running-stat drift between two checkpoints.")
    parser.add_argument("reference")
    parser.add_argument("candidate")
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--warn-ratio", type=float, default=0.02)
    args = parser.parse_args()

    ref = _state_dict(torch.load(args.reference, map_location="cpu"))
    cand = _state_dict(torch.load(args.candidate, map_location="cpu"))

    totals = defaultdict(lambda: [0.0, 0.0, 0])
    rows = []
    suffixes = ("running_mean", "running_var", "num_batches_tracked")
    for name, ref_value in ref.items():
        if not name.endswith(suffixes) or name not in cand:
            continue
        cand_value = cand[name]
        if not torch.is_tensor(ref_value) or not torch.is_tensor(cand_value) or ref_value.shape != cand_value.shape:
            continue
        delta = (cand_value.float() - ref_value.float()).norm().item()
        base = ref_value.float().norm().item()
        ratio = delta / (base + 1e-12)
        group = _group_name(name)
        totals[group][0] += delta
        totals[group][1] += base
        totals[group][2] += 1
        rows.append((ratio, delta, base, group, name))

    print("BN running-stat drift by group:")
    failed = False
    for group in sorted(totals):
        delta, base, count = totals[group]
        ratio = delta / (base + 1e-12)
        mark = " WARN" if ratio > args.warn_ratio else ""
        failed = failed or ratio > args.warn_ratio
        print(f"{group:18s} keys={count:3d} ratio={ratio:.6f} delta={delta:.6f} base={base:.6f}{mark}")

    print(f"\nTop {args.topk} BN buffer changes:")
    for ratio, delta, base, group, name in sorted(rows, reverse=True)[: args.topk]:
        print(f"{ratio:.6f} delta={delta:.6f} base={base:.6f} {group:18s} {name}")

    if failed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
