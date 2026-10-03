"""Bind supplied GPT review CSVs to existing initialization IDs; preserve media."""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


def read_csv(path):
    return list(csv.DictReader(path.open(encoding="utf-8-sig", newline="")))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--depthtrack", type=Path, required=True)
    parser.add_argument("--external", type=Path, required=True)
    parser.add_argument("--site", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    dt, external = read_csv(args.depthtrack), read_csv(args.external)
    assert len(dt) == 152 and len(external) == 1845
    source_rows = {"depthtrack": dt, "cdtb": [], "vot": []}
    for row in external:
        source_rows[row["dataset"]].append(row)
    receipt = {"version": "20261003-gpt-import", "human_confirmation_fabricated": False,
               "sources": [{"file": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                           for p in (args.depthtrack, args.external)], "datasets": {}}
    flat = []
    for dataset, rows in source_rows.items():
        path = args.site / "data" / f"{dataset}.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        key_name = "audit_id" if dataset == "depthtrack" else "case_id"
        by_id = {row[key_name]: row for row in rows}
        assert len(by_id) == len(rows) == manifest["count"]
        assert set(by_id) == {item["id"] for item in manifest["rows"]}
        for item in manifest["rows"]:
            row = by_id[item["id"]]
            if dataset == "depthtrack":
                assert item["category"] == row["generated_category"]
                assert item["attributes"] == [s.strip() for s in row["generated_attributes"].split("|") if s.strip()]
                review = dict(status=row["reviewed_category_status"],
                              category=row["reviewed_category_or_coarse_label"],
                              stable_attributes=row["supported_stable_attributes"],
                              uncertain_attributes=row["unobservable_attributes"],
                              conflicting_attributes=row["conflicting_attributes"],
                              context_only_notes=row["video_only_attributes"] + "\n" + row["multiframe_review_note"],
                              evidence=row["review_note"], evidence_frames=row["evidence_frames"],
                              initialization_observability=row["first_frame_category_observability"],
                              reviewer=row["reviewer"], review_date=row["review_date"])
            else:
                assert item["sequence"] == row["sequence"] and item["frame"] == row["init_frame"]
                assert item["number"] == int(row["anchor_number"])
                assert item["frame_index"] == int(row["init_frame_index"])
                assert item["category"] == row["provisional_category"]
                assert item["attributes"] == [s.strip() for s in row["provisional_attributes"].split(";") if s.strip()]
                assert row["human_status"] == "pending" and not row["human_reviewer"]
                review = dict(status=row["model_status"], category=row["model_category"],
                              stable_attributes=row["model_init_stable_attributes"],
                              uncertain_attributes=row["model_uncertain_attributes"],
                              conflicting_attributes="", context_only_notes=row["model_context_only_notes"],
                              evidence=row["model_evidence"], evidence_frames="", initialization_observability="",
                              reviewer=row["model_reviewer"], review_date="2026-10-02")
            assert review["status"] in {"supported", "conflicting", "corrected", "uncertain"}
            assert review["category"] and review["reviewer"] and review["evidence"]
            item["model_review"] = review
            flat.append(dict(dataset=dataset, anchor_number=item["number"], key=item["id"],
                             sequence=item["sequence"], init_frame=item["frame"],
                             **{"model_" + k: v for k, v in review.items()}, human_status="pending"))
        manifest["model_review_version"] = receipt["version"]
        path.write_text(json.dumps(manifest, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
        receipt["datasets"][dataset] = dict(count=len(rows), statuses=dict(Counter(item["model_review"]["status"] for item in manifest["rows"])), human_pending=len(rows))
    receipt["total"] = len(flat)
    (args.site / "data" / "model_review_receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (args.site / "data" / "gpt_initial_reviews.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == "__main__":
    main()
