"""Summarize only a completed, independently verified M81 result."""
from pathlib import Path
import hashlib, json, sys

result_path = Path(sys.argv[1])
audit_path = Path(sys.argv[2])
r = json.loads(result_path.read_text(encoding="utf-8"))
audit = json.loads(audit_path.read_text(encoding="utf-8"))
assert r["status"] == "complete_M81_t0_and_multistart_development"
assert audit["status"] == "complete_independent_scalar_M81_metric_and_receipt_verification"
assert hashlib.sha256(result_path.read_bytes()).hexdigest() == audit["result_sha256"]

pairs = [
    ("t0_category", "t0_native"),
    ("t0_category", "t0_empty_trained"),
    ("t0_category", "t0_empty_content"),
    ("t0_category", "t0_swapped"),
    ("multi_category", "multi_native"),
    ("multi_category", "multi_empty_trained"),
    ("multi_category", "multi_M78_category"),
    ("multi_empty_trained", "multi_M78_empty_trained"),
]
comparisons = []
for candidate, reference in pairs:
    a, b = r["aggregates"][candidate], r["aggregates"][reference]
    pa, pb = r["per_sequence"][candidate], r["per_sequence"][reference]
    assert set(pa) == set(pb) and len(pa) == 22
    rows = []
    leave_one_out = []
    for seq in pa:
        x, y = pa[seq], pb[seq]
        assert x["valid_frames"] == y["valid_frames"]
        rows.append(dict(sequence=seq, iou_delta_pp=100*(x["mean_iou"]-y["mean_iou"]),
                         H10_candidate=x["failure_episodes"], H10_reference=y["failure_episodes"],
                         low_delta=x["low_iou_frames"]-y["low_iou_frames"]))
        da = (a["iou_sum"]-x["iou_sum"])/(a["valid_frames"]-x["valid_frames"])
        db = (b["iou_sum"]-y["iou_sum"])/(b["valid_frames"]-y["valid_frames"])
        leave_one_out.append(dict(removed_sequence=seq, pooled_delta_pp=100*(da-db)))
    rows.sort(key=lambda x: x["iou_delta_pp"])
    comparisons.append(dict(candidate=candidate, reference=reference,
        pooled_delta_pp=100*(a["mean_iou"]-b["mean_iou"]),
        macro_delta_pp=100*(a["macro_sequence_mean_iou"]-b["macro_sequence_mean_iou"]),
        low_delta=a["low_iou_frames"]-b["low_iou_frames"],
        H10_delta=a["failure_episodes"]-b["failure_episodes"],
        worst_five=rows[:5], best_five=list(reversed(rows[-5:])),
        minimum_leave_one_out=min(leave_one_out,key=lambda x:x["pooled_delta_pp"])))
groups = ["primary_gates", "content_gates", "initialization_coverage_gates"]
report = dict(scope="Descriptive contrasts on reused Train development22; no comparison of H10 across protocols; no new promotion criteria",
              result_sha256=audit["result_sha256"], comparisons=comparisons,
              frozen_gates={g:dict(passed=sum(r[g].values()),total=len(r[g]),
                                   failed=[k for k,v in r[g].items() if not v]) for g in groups},
              broken_success_sequences=r["broken_success_sequences"])
print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
