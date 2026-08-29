#!/usr/bin/env python3
"""Reproduce the low22 protected/tentative transaction regression from artifacts.

The command exits with status 1 when the exact regression is present and status 0
when it is absent.  It intentionally reads only frozen evaluation artifacts, so it
is deterministic and runs in seconds without CUDA or the VOT dataset.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                value = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
            if isinstance(value, dict):
                yield value


def _metrics(side: Dict[str, Any]) -> Dict[str, float]:
    values = side.get("metrics_percent", {})
    return {name: float(values[name]) for name in ("eao", "acc", "rob")}


def _load_finalizer(path: Path):
    spec = importlib.util.spec_from_file_location("transaction_low22_finalizer", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import finalizer: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _collect_new_catastrophes(
    gate: Dict[str, Any],
    finalizer_path: Path,
    baseline_workspace: Optional[Path],
    baseline_tracker: Optional[str],
    candidate_workspace: Optional[Path],
    candidate_tracker: Optional[str],
) -> List[Dict[str, Any]]:
    embedded = gate.get("new_catastrophic_failure_anchors")
    if isinstance(embedded, list):
        return embedded
    required = (
        baseline_workspace,
        baseline_tracker,
        candidate_workspace,
        candidate_tracker,
    )
    if any(value is None for value in required):
        raise ValueError(
            "gate has no embedded catastrophic-anchor list; provide both workspaces "
            "and tracker ids"
        )
    finalizer = _load_finalizer(finalizer_path)
    baseline_outcomes, _, _, _ = finalizer.collect_confirmed_failure_outcomes(
        baseline_workspace, baseline_tracker
    )
    candidate_outcomes, _, _, _ = finalizer.collect_confirmed_failure_outcomes(
        candidate_workspace, candidate_tracker
    )
    if set(baseline_outcomes) != set(candidate_outcomes):
        raise RuntimeError("baseline/candidate anchor keys differ")
    output = []
    for key in sorted(baseline_outcomes):
        baseline = baseline_outcomes[key]
        candidate = candidate_outcomes[key]
        if candidate["failed"] and not baseline["failed"]:
            output.append(
                {
                    "anchor_key": key,
                    "sequence": candidate["sequence"],
                    "anchor": candidate["anchor"],
                    "direction": candidate["direction"],
                    "baseline_progress": baseline["progress"],
                    "candidate_progress": candidate["progress"],
                    "run_length": candidate["run_length"],
                }
            )
    return output


def _trace_stats(path: Path, candidate_progress: int) -> Dict[str, Any]:
    promotes_before_failure = 0
    state_conflicts_before_failure = 0
    frozen_state_starts_before_failure = 0
    protected_conflict_rejects_better_tentative = 0
    first_freeze = None

    for event in _iter_jsonl(path):
        frame_id = int(event.get("frame_id", -1))
        if frame_id < 0 or frame_id > candidate_progress:
            continue

        decision = event.get("decision") or {}
        if decision.get("action") == "promote":
            promotes_before_failure += 1

        if event.get("event_kind") == "state_conflict_candidate":
            state_conflicts_before_failure += 1
            if event.get("protected_bbox") == event.get("prior_bbox"):
                frozen_state_starts_before_failure += 1
                if first_freeze is None:
                    first_freeze = {
                        "frame_id": frame_id,
                        "prior_bbox": event.get("prior_bbox"),
                        "protected_bbox": event.get("protected_bbox"),
                        "writer_reasons": (event.get("writer_decision") or {}).get("reasons", []),
                    }

        reasons = set(decision.get("reasons") or [])
        protected_utility = decision.get("protected_utility")
        tentative_utility = decision.get("tentative_utility")
        if (
            decision.get("action") == "rollback"
            and "protected_hard_conflict" in reasons
            and isinstance(protected_utility, (int, float))
            and isinstance(tentative_utility, (int, float))
            and float(tentative_utility) > float(protected_utility)
        ):
            protected_conflict_rejects_better_tentative += 1

    return {
        "promotes_before_failure": promotes_before_failure,
        "state_conflicts_before_failure": state_conflicts_before_failure,
        "frozen_state_starts_before_failure": frozen_state_starts_before_failure,
        "protected_conflict_rejects_better_tentative": protected_conflict_rejects_better_tentative,
        "first_freeze": first_freeze,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--traces", type=Path, default=None)
    parser.add_argument(
        "--finalizer-script",
        type=Path,
        default=Path(__file__).with_name("finalize_vot_transaction_low22.py"),
    )
    parser.add_argument("--baseline-workspace", type=Path)
    parser.add_argument("--baseline-tracker")
    parser.add_argument("--candidate-workspace", type=Path)
    parser.add_argument("--candidate-tracker")
    parser.add_argument("--max-examples", type=int, default=12)
    args = parser.parse_args()

    gate = _load_json(args.gate)
    diagnostics = _load_json(args.diagnostics)
    summary = diagnostics["transaction_trace_summary"]
    totals = summary["totals"]
    traces_root = args.traces or Path(summary["trace_root"])

    baseline = _metrics(gate["baseline"])
    candidate = _metrics(gate["candidate"])
    failure_delta = int(gate["failure_delta"])
    events_started = int(totals["events_started"])
    state_conflicts = int(totals["state_conflict_candidates"])
    template_candidates = int(totals["template_candidates"])
    state_conflict_fraction = state_conflicts / max(events_started, 1)

    catastrophic_anchors = _collect_new_catastrophes(
        gate,
        args.finalizer_script,
        args.baseline_workspace,
        args.baseline_tracker,
        args.candidate_workspace,
        args.candidate_tracker,
    )

    examples: List[Dict[str, Any]] = []
    missing_traces: List[str] = []
    for item in catastrophic_anchors:
        sequence = str(item["sequence"])
        anchor = int(item["anchor"])
        trace_path = traces_root / f"{sequence}__anchor-{anchor:06d}.jsonl"
        if not trace_path.is_file():
            missing_traces.append(str(trace_path))
            continue
        stats = _trace_stats(trace_path, int(item["candidate_progress"]))
        examples.append(
            {
                "anchor_key": item["anchor_key"],
                "baseline_progress": int(item["baseline_progress"]),
                "candidate_progress": int(item["candidate_progress"]),
                **stats,
            }
        )

    zero_promote_catastrophes = [
        item for item in examples if item["promotes_before_failure"] == 0
    ]
    frozen_zero_promote_catastrophes = [
        item
        for item in zero_promote_catastrophes
        if item["frozen_state_starts_before_failure"] > 0
    ]
    better_tentative_rejected = sum(
        int(item["protected_conflict_rejects_better_tentative"]) for item in examples
    )

    checks = {
        "complete_303_anchor_scope": (
            gate.get("status") == "complete"
            and int(gate.get("scope", {}).get("anchor_count", gate.get("anchor_count", -1)))
            == 303
        ),
        "eao_regressed": candidate["eao"] < baseline["eao"],
        "rob_regressed": candidate["rob"] < baseline["rob"],
        "confirmed_failures_increased": failure_delta > 0,
        "state_conflicts_dominate": (
            state_conflicts > template_candidates and state_conflict_fraction >= 0.90
        ),
        "new_catastrophe_without_template_promote": bool(zero_promote_catastrophes),
        "new_catastrophe_after_state_freeze": bool(frozen_zero_promote_catastrophes),
        "better_tentative_rejected_on_protected_conflict": better_tentative_rejected > 0,
        "all_catastrophic_traces_present": not missing_traces,
    }
    regression_reproduced = all(checks.values())

    output = {
        "schema": "sutrack_transaction_regression_diagnostic_v1",
        "verdict": "REGRESSION_REPRODUCED" if regression_reproduced else "REGRESSION_NOT_REPRODUCED",
        "baseline_percent": baseline,
        "candidate_percent": candidate,
        "failure_delta": failure_delta,
        "transaction_counts": {
            "events_started": events_started,
            "state_conflict_candidates": state_conflicts,
            "template_candidates": template_candidates,
            "state_conflict_fraction": state_conflict_fraction,
            "promotes": int(totals["promotes"]),
            "rollbacks": int(totals["rollbacks"]),
        },
        "new_catastrophic_anchors": len(catastrophic_anchors),
        "catastrophes_without_promote_before_failure": len(zero_promote_catastrophes),
        "catastrophes_with_frozen_state_before_failure": len(frozen_zero_promote_catastrophes),
        "better_tentative_rejected_count": better_tentative_rejected,
        "checks": checks,
        "missing_traces": missing_traces,
        "examples": frozen_zero_promote_catastrophes[: max(args.max_examples, 0)],
    }
    print(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True))
    return 1 if regression_reproduced else 0


if __name__ == "__main__":
    raise SystemExit(main())
