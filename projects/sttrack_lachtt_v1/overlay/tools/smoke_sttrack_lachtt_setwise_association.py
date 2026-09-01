#!/usr/bin/env python3
"""Engineering smoke for permutation-equivariant setwise association."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from lib.models.sttrack.lachtt_setwise_association import (
    SetwiseCandidateAssociation,
    setwise_association_loss,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True,
                      allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def maximum_permutation_error(first, second, permutation, candidate_valid):
    errors = []
    for key in ("beneficial_logits", "catastrophic_logits", "gain"):
        errors.append((first[key][:, permutation] - second[key]).abs().max())
    selection_error = (first["selection_logits"][:, :-1][:, permutation] -
                       second["selection_logits"][:, :-1]).abs()
    errors.append(selection_error[candidate_valid[:, permutation]].max())
    errors.append((first["selection_logits"][:, -1] -
                   second["selection_logits"][:, -1]).abs().max())
    return float(torch.stack(errors).max().item())


def main():
    args = parse_args()
    args.output = args.output.resolve()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.manual_seed(2026)
    torch.cuda.manual_seed_all(2026)
    batch, horizon, candidates = 4, 4, 6
    features = torch.randn(batch, horizon, candidates, 128, device="cuda")
    scalars = torch.randn(batch, horizon, candidates, 9, device="cuda")
    valid = torch.ones(batch, candidates, dtype=torch.bool, device="cuda")
    valid[-1, -1] = False
    gains = torch.tensor([
        [0.32, -0.40, 0.04, 0.11, -0.08, 0.01],
        [-0.03, 0.00, -0.07, 0.02, -0.12, -0.01],
        [0.08, 0.26, -0.31, 0.04, 0.12, -0.02],
        [-0.28, -0.11, 0.01, -0.44, 0.00, 0.20],
    ], device="cuda")
    beneficial = gains >= 0.05
    catastrophic = gains <= -0.25
    model = SetwiseCandidateAssociation().cuda()
    parameter_count = sum(value.numel() for value in model.parameters())
    model.eval()
    with torch.no_grad():
        original = model(features, scalars, valid)
        permutation = torch.tensor([2, 0, 5, 1, 4, 3], device="cuda")
        permuted = model(features[:, :, permutation],
                         scalars[:, :, permutation], valid[:, permutation])
    permutation_error = maximum_permutation_error(
        original, permuted, permutation, valid)
    if permutation_error > 2e-6:
        raise RuntimeError("setwise permutation equivariance failed")

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3,
                                  weight_decay=1e-4)
    before = {name: value.detach().clone()
              for name, value in model.named_parameters()}
    traces = []
    for step in range(4):
        outputs = model(features, scalars, valid)
        losses = setwise_association_loss(
            outputs, gains, beneficial, catastrophic, valid)
        optimizer.zero_grad(set_to_none=True)
        losses["total"].backward()
        gradient = float(torch.nn.utils.clip_grad_norm_(
            model.parameters(), 5.0).item())
        optimizer.step()
        traces.append({"step": step, "gradient_norm": gradient,
                       **{key: float(value.detach().item())
                          for key, value in losses.items()}})
    changed = sum(not torch.equal(before[name], value.detach())
                  for name, value in model.named_parameters())
    if (changed == 0 or traces[-1]["total"] >= traces[0]["total"] or
            any(not math.isfinite(value) for row in traces
                for key, value in row.items() if key != "step")):
        raise RuntimeError("setwise engineering update failed")
    source = (REPOSITORY_ROOT /
              "lib/models/sttrack/lachtt_setwise_association.py")
    runner = Path(__file__).resolve()
    result = {
        "schema": "sttrack-lachtt-m4-setwise-engineering-smoke/v1",
        "complete": True,
        "accepted": True,
        "scientific_scope": "engineering invariance/backward smoke only",
        "repository_commit": subprocess.check_output(
            ["git", "-C", str(REPOSITORY_ROOT), "rev-parse", "HEAD"],
            text=True).strip(),
        "parameter_count": parameter_count,
        "batch": batch,
        "horizon": horizon,
        "candidates": candidates,
        "permutation_error_max": permutation_error,
        "changed_parameter_tensors": changed,
        "traces": traces,
        "source_sha256": sha256_file(source),
        "runner_sha256": sha256_file(runner),
        "checkpoint_written": False,
        "tracking_run": False,
        "qwen_used": False,
        "vot_run": False,
        "automatic_next_stage": False,
    }
    args.output.mkdir(parents=True)
    atomic_json(args.output / "result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
