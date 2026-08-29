#!/usr/bin/env python3
"""Diagnose low22 outcome changes caused by template-only transactions.

This artifact-only differential is intentionally deterministic and CUDA-free.
It exits with status 1 while the candidate has the exact formal regression
under investigation (EAO/ROB lower and confirmed failures higher than the
frozen identity-only baseline), and status 0 otherwise.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                value = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSONL at {path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(value, dict):
                raise ValueError(f"expected JSON object at {path}:{line_number}")
            yield value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("transaction_low22_finalizer", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import finalizer: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def metric_percent(gate: Dict[str, Any], side: str) -> Dict[str, float]:
    values = gate[side]["metrics_percent"]
    return {name: float(values[name]) for name in ("eao", "acc", "rob")}


def compact_event(event: Dict[str, Any]) -> Dict[str, Any]:
    decision = event.get("decision") or {}
    protected = event.get("protected_evidence") or {}
    tentative = event.get("tentative_evidence") or {}
    writer = event.get("writer_decision") or {}
    return {
        "frame_id": int(event.get("frame_id", -1)),
        "event_kind": event.get("event_kind"),
        "action": decision.get("action"),
        "event_id": decision.get("event_id"),
        "reasons": list(decision.get("reasons") or []),
        "protected_utility": decision.get("protected_utility"),
        "tentative_utility": decision.get("tentative_utility"),
        "protected_confidence": protected.get("confidence"),
        "tentative_confidence": tentative.get("confidence"),
        "protected_identity": protected.get("identity_similarity"),
        "tentative_identity": tentative.get("identity_similarity"),
        "writer_identity": writer.get("identity_similarity"),
        "writer_reasons": list(writer.get("reasons") or []),
    }


def trace_before_progress(path: Path, progress: int) -> Dict[str, Any]:
    starts: List[Dict[str, Any]] = []
    promotes: List[Dict[str, Any]] = []
    rollbacks: List[Dict[str, Any]] = []
    holds = 0
    state_conflicts = 0
    all_events = 0
    for event in iter_jsonl(path):
        if event.get("type") != "transaction_frame":
            continue
        frame_id = int(event.get("frame_id", -1))
        if frame_id < 0 or frame_id > progress:
            continue
        all_events += 1
        if event.get("event_kind") == "state_conflict_candidate":
            state_conflicts += 1
        compact = compact_event(event)
        if event.get("event_kind") == "template_candidate":
            starts.append(compact)
        action = (event.get("decision") or {}).get("action")
        if action == "promote":
            promotes.append(compact)
        elif action == "rollback":
            rollbacks.append(compact)
        elif action == "hold":
            holds += 1
    return {
        "trace_sha256": sha256_file(path),
        "transaction_frames_before_outcome": all_events,
        "template_starts_before_outcome": len(starts),
        "promotes_before_outcome": len(promotes),
        "rollbacks_before_outcome": len(rollbacks),
        "holds_before_outcome": holds,
        "state_conflicts_before_outcome": state_conflicts,
        "first_start": starts[0] if starts else None,
        "first_promote": promotes[0] if promotes else None,
        "last_promote": promotes[-1] if promotes else None,
        "last_rollback": rollbacks[-1] if rollbacks else None,
        "frames_from_last_promote_to_outcome": (
            progress - int(promotes[-1]["frame_id"]) if promotes else None
        ),
    }


def transition_record(
    baseline: Dict[str, Any],
    candidate: Dict[str, Any],
    trace_root: Path,
) -> Dict[str, Any]:
    sequence = str(candidate["sequence"])
    anchor = int(candidate["anchor"])
    trace_path = trace_root / f"{sequence}__anchor-{anchor:06d}.jsonl"
    if not trace_path.is_file():
        raise FileNotFoundError(trace_path)
    outcome_progress = (
        int(candidate["progress"])
        if candidate["failed"]
        else int(candidate["run_length"])
    )
    return {
        "anchor_key": candidate["anchor_key"],
        "sequence": sequence,
        "anchor": anchor,
        "direction": candidate["direction"],
        "run_length": int(candidate["run_length"]),
        "baseline_failed": bool(baseline["failed"]),
        "baseline_progress": int(baseline["progress"]),
        "candidate_failed": bool(candidate["failed"]),
        "candidate_progress": int(candidate["progress"]),
        "progress_delta": int(candidate["progress"]) - int(baseline["progress"]),
        "trace": trace_before_progress(trace_path, outcome_progress),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--finalizer-script", type=Path, required=True)
    parser.add_argument("--baseline-workspace", type=Path, required=True)
    parser.add_argument("--baseline-tracker", required=True)
    parser.add_argument("--candidate-workspace", type=Path, required=True)
    parser.add_argument("--candidate-tracker", required=True)
    parser.add_argument("--max-both-failed-examples", type=int, default=12)
    args = parser.parse_args()

    gate = load_json(args.gate)
    diagnostics = load_json(args.diagnostics)
    finalizer = load_module(args.finalizer_script)

    baseline_outcomes, baseline_failures, _, _ = (
        finalizer.collect_confirmed_failure_outcomes(
            args.baseline_workspace, args.baseline_tracker
        )
    )
    candidate_outcomes, candidate_failures, _, _ = (
        finalizer.collect_confirmed_failure_outcomes(
            args.candidate_workspace, args.candidate_tracker
        )
    )
    if set(baseline_outcomes) != set(candidate_outcomes):
        raise RuntimeError("baseline and candidate anchor keys differ")
    if len(baseline_outcomes) != 303:
        raise RuntimeError("expected exactly 303 low22 anchors")
    if baseline_failures != int(gate["baseline"]["confirmed_failures"]):
        raise RuntimeError("baseline failure count disagrees with gate")
    if candidate_failures != int(gate["candidate"]["confirmed_failures"]):
        raise RuntimeError("candidate failure count disagrees with gate")

    new_catastrophes: List[Dict[str, Any]] = []
    rescues: List[Dict[str, Any]] = []
    both_failed_earlier: List[Dict[str, Any]] = []
    both_failed_delayed: List[Dict[str, Any]] = []
    transition_counts: Counter[str] = Counter()
    sequence_counts: Dict[str, Counter[str]] = defaultdict(Counter)

    for key in sorted(baseline_outcomes):
        baseline = baseline_outcomes[key]
        candidate = candidate_outcomes[key]
        if not baseline["failed"] and candidate["failed"]:
            kind = "new_catastrophe"
            record = transition_record(baseline, candidate, args.traces)
            new_catastrophes.append(record)
        elif baseline["failed"] and not candidate["failed"]:
            kind = "rescue"
            record = transition_record(baseline, candidate, args.traces)
            rescues.append(record)
        elif baseline["failed"] and candidate["failed"]:
            delta = int(candidate["progress"]) - int(baseline["progress"])
            if delta < 0:
                kind = "both_failed_earlier"
                both_failed_earlier.append(
                    transition_record(baseline, candidate, args.traces)
                )
            elif delta > 0:
                kind = "both_failed_delayed"
                both_failed_delayed.append(
                    transition_record(baseline, candidate, args.traces)
                )
            else:
                kind = "both_failed_same_progress"
        else:
            kind = "both_survived"
        transition_counts[kind] += 1
        sequence_counts[str(candidate["sequence"])][kind] += 1

    baseline_metrics = metric_percent(gate, "baseline")
    candidate_metrics = metric_percent(gate, "candidate")
    totals = diagnostics["transaction_trace_summary"]["totals"]
    trace_files = sorted(args.traces.glob("*.jsonl"))
    if len(trace_files) != 303:
        raise RuntimeError("expected exactly 303 transaction traces")

    regression_present = (
        candidate_metrics["eao"] < baseline_metrics["eao"]
        and candidate_metrics["rob"] < baseline_metrics["rob"]
        and candidate_failures > baseline_failures
        and len(new_catastrophes) > len(rescues)
    )
    output = {
        "schema": "sutrack_template_transaction_outcome_diagnostic/v1",
        "verdict": (
            "TEMPLATE_TRANSACTION_REGRESSION_PRESENT"
            if regression_present
            else "TEMPLATE_TRANSACTION_REGRESSION_ABSENT"
        ),
        "formal_scope": {
            "sequence_count": int(gate["sequence_count"]),
            "anchor_count": int(gate["anchor_count"]),
            "toolkit": gate["toolkit"],
            "automatic_full127_launch": bool(gate["automatic_full127_launch"]),
        },
        "metrics_percent": {
            "baseline": baseline_metrics,
            "candidate": candidate_metrics,
            "delta": {
                name: candidate_metrics[name] - baseline_metrics[name]
                for name in ("eao", "acc", "rob")
            },
        },
        "failures": {
            "baseline": baseline_failures,
            "candidate": candidate_failures,
            "delta": candidate_failures - baseline_failures,
        },
        "transition_counts": dict(sorted(transition_counts.items())),
        "sequence_transition_counts": {
            sequence: dict(sorted(counts.items()))
            for sequence, counts in sorted(sequence_counts.items())
        },
        "template_event_totals": {
            key: int(totals[key])
            for key in (
                "events_started",
                "template_candidates",
                "state_conflict_candidates",
                "promotes",
                "rollbacks",
                "unresolved_at_trajectory_end",
                "creation_errors",
                "recoverable_errors",
            )
        },
        "new_catastrophes_after_promote": sum(
            item["trace"]["promotes_before_outcome"] > 0
            for item in new_catastrophes
        ),
        "new_catastrophes_without_promote": sum(
            item["trace"]["promotes_before_outcome"] == 0
            for item in new_catastrophes
        ),
        "rescues_after_promote": sum(
            item["trace"]["promotes_before_outcome"] > 0 for item in rescues
        ),
        "new_catastrophes": new_catastrophes,
        "rescues": rescues,
        "both_failed_earlier_examples": sorted(
            both_failed_earlier, key=lambda item: item["progress_delta"]
        )[: max(args.max_both_failed_examples, 0)],
        "both_failed_delayed_examples": sorted(
            both_failed_delayed,
            key=lambda item: item["progress_delta"],
            reverse=True,
        )[: max(args.max_both_failed_examples, 0)],
        "artifact_sha256": {
            "gate": sha256_file(args.gate),
            "diagnostics": sha256_file(args.diagnostics),
            "finalizer_script": sha256_file(args.finalizer_script),
        },
    }
    print(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True))
    return 1 if regression_present else 0


if __name__ == "__main__":
    raise SystemExit(main())
