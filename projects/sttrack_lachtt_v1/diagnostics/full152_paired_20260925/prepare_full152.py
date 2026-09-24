"""Prepare paired M67/M82 Full152 training without changing the original runs."""

import copy
import hashlib
import json
from pathlib import Path

import torch


BASE = Path("/root/autodl-tmp")
OLD67 = BASE / "sttrack_m67_supervised_semantic_support_20260907"
OLD82 = BASE / "sttrack_m82_native_preservation_20260909"
PAIR = BASE / "sttrack_full152_paired_20260925"
ZERO = OLD82 / "native_parity/category_zero.pth"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def link(target, path):
    path.symlink_to(target, target_is_directory=target.is_dir())


def bank():
    fit = torch.load(OLD67 / "text_fit.pt", map_location="cpu")
    dev = torch.load(OLD67 / "text_development.pt", map_location="cpu")
    assert torch.equal(fit["empty"], dev["empty"])
    assert fit["encoder_sha256"] == dev["encoder_sha256"]
    assert fit["lexical_policy"] == dev["lexical_policy"]
    assert not set(fit["sequences"]) & set(dev["sequences"])
    merged = dict(
        sequences=fit["sequences"] + dev["sequences"],
        tokens=torch.cat((fit["tokens"], dev["tokens"]), dim=0),
        mask=torch.cat((fit["mask"], dev["mask"]), dim=0),
        empty=fit["empty"],
        encoder_sha256=fit["encoder_sha256"],
        lexical_policy=fit["lexical_policy"],
        original_initialization_phrases=fit["original_initialization_phrases"]
        + dev["original_initialization_phrases"],
        source_fit_sha256=sha(OLD67 / "text_fit.pt"),
        source_development_sha256=sha(OLD67 / "text_development.pt"),
    )
    assert len(merged["sequences"]) == 152
    path = PAIR / "text_full152.pt"
    torch.save(merged, path)
    return path


def prepare_arm(name, old, bank_path, rows, original, zero_sha):
    root = PAIR / name
    root.mkdir()
    (root / "native_parity").mkdir()
    (root / "training").mkdir()
    link(old / "code", root / "code")
    link(old / "integration.json", root / "integration.json")
    link(ZERO, root / "native_parity" / ("control_zero.pth" if name == "M67" else "category_zero.pth"))
    sources = (
        ["causal_training.py", "support_loss.py"]
        if name == "M67"
        else ["causal_training.py", "support_loss.py", "window_competition.py", "native_preservation.py"]
    )
    for filename in sources:
        link(old / filename, root / filename)
    source = (old / "train_causal.py").read_text()
    assert f"root = Path('{old}')" in source
    (root / "train_causal.py").write_text(
        source.replace(f"root = Path('{old}')", f"root = Path('{root}')")
    )
    spec = dict(
        experiment="Full152 paired M67/M82",
        status="prepared",
        arm=name,
        seed=2027,
        dataset_root=original["dataset_root"],
        sequence_order=rows,
        total_training_image_frames=sum(row["rgb_frames"] for row in rows),
        total_training_track_calls=sum(row["rgb_frames"] - 1 for row in rows),
        maximum_optimizer_steps=sum((row["rgb_frames"] - 1 + 31) // 32 for row in rows),
        gradient_accumulation_frames=original["gradient_accumulation_frames"],
        learning_rate=original["learning_rate"],
        weight_decay=original["weight_decay"],
        gradient_clip=original["gradient_clip"],
        native_checkpoint=original["native_checkpoint"],
        native_checkpoint_sha256=original["native_checkpoint_sha256"],
        initial_checkpoint_sha256={
            ("control" if name == "M67" else "category"): zero_sha
        },
        integration_sha256=sha(root / "integration.json"),
        causal_script_sha256=sha(root / "causal_training.py"),
        support_loss_sha256=sha(root / "support_loss.py"),
        training_script_sha256=sha(root / "train_causal.py"),
        support_loss_weights={("control" if name == "M67" else "category"): 0.0},
        source_training_spec_sha256=sha(old / "training_spec.json"),
        source_fit_bank_sha256=sha(OLD67 / "text_fit.pt"),
        source_development_bank_sha256=sha(OLD67 / "text_development.pt"),
        selection_protocol="one full pass, final checkpoint; no development-22 selection",
        text_protocol="old frozen target-marked full image plus target crop; category slot0, empty attributes",
    )
    if name == "M67":
        link(bank_path, root / "text_fit.pt")
        spec["text_fit_sha256"] = sha(bank_path)
    else:
        spec["banks"] = {"fit": {"category": {"path": str(bank_path), "sha256": sha(bank_path)}}}
        spec["window_loss_sha256"] = sha(root / "window_competition.py")
        spec["preservation_loss_sha256"] = sha(root / "native_preservation.py")
        spec["preservation_weight"] = original["preservation_weight"]
    assert spec["total_training_image_frames"] == 219954
    assert spec["total_training_track_calls"] == 219802
    write_json(root / "training_spec.json", spec)
    if name == "M82":
        write_json(root / "frozen.json", {"training_spec_sha256": sha(root / "training_spec.json")})
    return dict(root=str(root), spec_sha256=sha(root / "training_spec.json"),
                script_sha256=spec["training_script_sha256"], initial_sha256=zero_sha)


def main():
    PAIR.mkdir()
    old67 = json.loads((OLD67 / "training_spec.json").read_text())
    old82 = json.loads((OLD82 / "training_spec.json").read_text())
    assert old67["sequence_order"] == old82["sequence_order"]
    inventory = json.loads((OLD67 / "data_inventory.json").read_text())
    dev = {row["sequence"]: row for row in inventory["sequences_detail"]
           if row["split"] == "development"}
    assert set(dev) == set(old67["development_sequences"])
    rows = copy.deepcopy(old67["sequence_order"])
    for sequence in old67["development_sequences"]:
        row = copy.deepcopy(dev[sequence])
        row["split"] = "fit"
        row["old_fit_sequence"] = False
        rows.append(row)
    assert len(rows) == 152 and len({row["sequence"] for row in rows}) == 152
    bank_path = bank()
    merged = torch.load(bank_path, map_location="cpu")
    assert set(merged["sequences"]) == {row["sequence"] for row in rows}
    zero = torch.load(ZERO, map_location="cpu")
    assert zero["architecture"] == "semantic_spatial_support_v1"
    assert zero["null_support"] and zero["use_text"]
    zero_sha = sha(ZERO)
    results = [
        prepare_arm("M67", OLD67, bank_path, rows, old67, zero_sha),
        prepare_arm("M82", OLD82, bank_path, rows, old82, zero_sha),
    ]
    write_json(PAIR / "preparation.json", dict(
        status="prepared", seed=2027, sequence_count=152,
        total_training_track_calls=219802, bank_sha256=sha(bank_path),
        shared_initial_adapter_sha256=zero_sha, arms=results,
        old_development_22_now_training=True,
    ))
    print((PAIR / "preparation.json").read_text())


if __name__ == "__main__":
    main()
