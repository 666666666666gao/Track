#!/usr/bin/env python3
"""Sequence-disjoint nested OOF selector for STTrack LACH-TT branches."""

import argparse
from collections import defaultdict
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


MODEL_SEEDS = (2026, 2027, 2028)
OUTER_FOLDS = 6
INNER_FOLDS = 5
EPOCHS = 20
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
THRESHOLDS = np.linspace(0.01, 0.999, 990).tolist()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path):
    path = Path(path).resolve()
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": sha256_file(path)}


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2,
                      sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_gzip_jsonl(path, rows):
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    os.close(descriptor)
    try:
        with gzip.open(temporary, "wt", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, sort_keys=True,
                                        allow_nan=False) + "\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def stable_fold(sequence, salt, folds):
    value = hashlib.sha256((salt + "\0" + sequence).encode()).digest()
    return int.from_bytes(value[:8], "big") % folds


class AssociationSelector(nn.Module):
    """Modality-separated temporal encoding plus candidate distractor attention."""

    def __init__(self):
        super().__init__()
        projection = 24
        hidden = 96
        names = ("native_rgb", "native_depth", "native_fused",
                 "clip_image", "query_rgb", "query_depth")
        self.projections = nn.ModuleDict({
            name: nn.Sequential(nn.Linear(768, projection), nn.GELU(),
                                nn.LayerNorm(projection)) for name in names
        })
        self.depth_encoder = nn.Sequential(
            nn.Conv2d(2, 8, 3, padding=1), nn.GELU(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d(1))
        self.scalar_encoder = nn.Sequential(
            nn.Linear(15, 32), nn.GELU(), nn.Linear(32, 16),
            nn.LayerNorm(16))
        input_size = projection * len(names) + 16 + 16 + 2
        self.fusion = nn.Sequential(
            nn.Linear(input_size, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.temporal = nn.GRU(hidden, hidden, batch_first=True)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=4, dim_feedforward=192,
            dropout=0.10, activation="gelu", batch_first=True,
            norm_first=True)
        self.distractor = nn.TransformerEncoder(layer, num_layers=1)
        self.beneficial_head = nn.Linear(hidden, 1)
        self.catastrophic_head = nn.Linear(hidden, 1)
        self.gain_head = nn.Linear(hidden, 1)

    def forward(self, features, anchor_image, anchor_text):
        # Every dense feature is BxAgexCandidatexD.
        parts = [self.projections[name](features[name].float())
                 for name in self.projections]
        batch, ages, candidates = features["clip_image"].shape[:3]
        depth = features["raw_depth"].float().reshape(
            batch * ages * candidates, 2, 16, 16)
        depth = self.depth_encoder(depth).reshape(
            batch, ages, candidates, 16)
        scalars = torch.asinh(features["scalars"].float())
        scalars = self.scalar_encoder(scalars)
        clip_image = F.normalize(features["clip_image"].float(), dim=-1)
        init = F.normalize(anchor_image.float(), dim=-1)[:, None, None, :]
        text = F.normalize(anchor_text.float(), dim=-1)[:, None, None, :]
        similarities = torch.stack([
            (clip_image * init).sum(dim=-1),
            (clip_image * text).sum(dim=-1)], dim=-1)
        fused = self.fusion(torch.cat(parts + [depth, scalars, similarities],
                                      dim=-1))
        temporal_input = fused.permute(0, 2, 1, 3).reshape(
            batch * candidates, ages, -1)
        _, state = self.temporal(temporal_input)
        candidate_state = state[-1].reshape(batch, candidates, -1)
        associated = self.distractor(candidate_state)
        return {
            "beneficial_logit": self.beneficial_head(associated).squeeze(-1),
            "catastrophic_logit": self.catastrophic_head(associated).squeeze(-1),
            "gain": torch.tanh(self.gain_head(associated).squeeze(-1)),
        }


def parameter_count(model):
    return sum(parameter.numel() for parameter in model.parameters())


def load_labels(path):
    grouped = defaultdict(dict)
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            grouped[(row["sequence"], int(row["trigger_frame"]))][
                row["branch_id"]] = row
    return grouped


def load_data(spec, smoke=False):
    root = Path(spec["collection_root"])
    labels = load_labels(Path(spec["gate_a_root"]) /
                         "labeled_actions.jsonl.gz")
    feature_lists = defaultdict(list)
    anchor_images, anchor_texts = [], []
    beneficial, catastrophic, gains = [], [], []
    sequences, triggers, branch_names = [], [], []
    anchor_cache = {}
    limit = 24 if smoke else None
    for shard in (0, 1):
        shard_root = root / ("shard%d" % shard)
        with (shard_root / "events.jsonl").open("r", encoding="utf-8") as stream:
            for line in stream:
                event = json.loads(line)
                key = (event["sequence"], int(event["trigger_frame"]))
                event_labels = labels.get(key)
                if not event_labels:
                    raise ValueError("event label join is incomplete")
                names = [row["name"] for row in
                         event["trajectory"][0]["branches"]]
                ordered = [event_labels[name] for name in names]
                if all(row["label"] == "unavailable" for row in ordered):
                    continue
                if any(row["label"] == "unavailable" for row in ordered):
                    raise ValueError("partially unavailable event")
                values = torch.load(shard_root / event["feature_path"],
                                    map_location="cpu")
                for name, value in values.items():
                    feature_lists[name].append(value)
                sequence = event["sequence"]
                if sequence not in anchor_cache:
                    anchor_cache[sequence] = torch.load(
                        shard_root / event["anchor_path"], map_location="cpu")
                anchor_images.append(anchor_cache[sequence]["initial_image"][0])
                anchor_texts.append(anchor_cache[sequence]["identity_text"][0])
                beneficial.append(torch.tensor([
                    row["label"] == "beneficial" for row in ordered],
                    dtype=torch.float32))
                catastrophic.append(torch.tensor([
                    row["label"] == "catastrophic" for row in ordered],
                    dtype=torch.float32))
                gains.append(torch.tensor([
                    row["mean_iou_gain"] for row in ordered],
                    dtype=torch.float32))
                sequences.append(sequence)
                triggers.append(key[1])
                branch_names.append(names)
                if limit is not None and len(sequences) >= limit:
                    break
        if limit is not None and len(sequences) >= limit:
            break
    features = {name: torch.stack(values, dim=0)
                for name, values in feature_lists.items()}
    data = {
        "features": features,
        "anchor_image": torch.stack(anchor_images),
        "anchor_text": torch.stack(anchor_texts),
        "beneficial": torch.stack(beneficial),
        "catastrophic": torch.stack(catastrophic),
        "gain": torch.stack(gains),
        "sequences": sequences,
        "triggers": triggers,
        "branch_names": branch_names,
    }
    if not sequences or any(value.shape[0] != len(sequences)
                            for value in features.values()):
        raise ValueError("empty or misaligned selector data")
    return data


def move_numeric_data(data, device):
    result = dict(data)
    for name in ("anchor_image", "anchor_text", "beneficial",
                 "catastrophic", "gain"):
        result[name] = data[name].to(device)
    result["features"] = {
        name: value.to(device) for name, value in data["features"].items()}
    return result


def batch_from(data, indexes):
    index = torch.as_tensor(indexes, device=data["beneficial"].device,
                            dtype=torch.long)
    return ({name: value.index_select(0, index)
             for name, value in data["features"].items()},
            data["anchor_image"].index_select(0, index),
            data["anchor_text"].index_select(0, index),
            data["beneficial"].index_select(0, index),
            data["catastrophic"].index_select(0, index),
            data["gain"].index_select(0, index))


def train_model(data, train_indexes, seed, epochs=EPOCHS):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = AssociationSelector().cuda()
    if parameter_count(model) > 400000:
        raise RuntimeError("selector exceeds frozen 400k parameter cap")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    y_b = data["beneficial"][train_indexes]
    y_c = data["catastrophic"][train_indexes]
    b_pos = max(1.0, float(y_b.sum().item()))
    c_pos = max(1.0, float(y_c.sum().item()))
    b_weight = torch.tensor(
        max(1.0, (y_b.numel() - b_pos) / b_pos), device="cuda")
    c_weight = torch.tensor(
        max(1.0, (y_c.numel() - c_pos) / c_pos), device="cuda")
    generator = np.random.RandomState(seed)
    first_loss, last_loss = None, None
    for epoch in range(epochs):
        order = np.asarray(train_indexes)[generator.permutation(len(train_indexes))]
        model.train()
        epoch_losses = []
        for offset in range(0, len(order), BATCH_SIZE):
            indexes = order[offset:offset + BATCH_SIZE]
            features, anchor_image, anchor_text, target_b, target_c, target_gain = batch_from(
                data, indexes)
            output = model(features, anchor_image, anchor_text)
            benefit_loss = F.binary_cross_entropy_with_logits(
                output["beneficial_logit"], target_b, pos_weight=b_weight)
            catastrophe_loss = F.binary_cross_entropy_with_logits(
                output["catastrophic_logit"], target_c, pos_weight=c_weight)
            gain_loss = F.smooth_l1_loss(output["gain"], target_gain)
            utility_logit = (output["beneficial_logit"] -
                             1.5 * F.softplus(output["catastrophic_logit"]))
            positive_events = target_b.sum(dim=1) > 0
            if positive_events.any():
                targets = target_gain[positive_events].argmax(dim=1)
                listwise = F.cross_entropy(
                    utility_logit[positive_events], targets)
            else:
                listwise = utility_logit.sum() * 0.0
            catastrophe_selection = (
                F.softmax(utility_logit, dim=1) * target_c).sum(dim=1).mean()
            loss = (benefit_loss + 1.5 * catastrophe_loss +
                    0.5 * gain_loss + 0.5 * listwise +
                    1.0 * catastrophe_selection)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(float(loss.item()))
        mean_loss = float(np.mean(epoch_losses))
        if first_loss is None:
            first_loss = mean_loss
        last_loss = mean_loss
    return model, {"first_epoch_loss": first_loss,
                   "last_epoch_loss": last_loss,
                   "positive_weight": float(b_weight.item()),
                   "catastrophic_weight": float(c_weight.item())}


def predict(model, data, indexes):
    model.eval()
    results = []
    with torch.no_grad():
        for offset in range(0, len(indexes), BATCH_SIZE):
            batch_indexes = indexes[offset:offset + BATCH_SIZE]
            features, anchor_image, anchor_text, target_b, target_c, target_gain = batch_from(
                data, batch_indexes)
            output = model(features, anchor_image, anchor_text)
            benefit = torch.sigmoid(output["beneficial_logit"])
            catastrophe = torch.sigmoid(output["catastrophic_logit"])
            utility = benefit * (1.0 - catastrophe)
            for local, global_index in enumerate(batch_indexes):
                results.append({
                    "index": int(global_index),
                    "utility": utility[local].cpu().tolist(),
                    "beneficial_probability": benefit[local].cpu().tolist(),
                    "catastrophic_probability": catastrophe[local].cpu().tolist(),
                    "predicted_gain": output["gain"][local].cpu().tolist(),
                    "target_beneficial": target_b[local].cpu().tolist(),
                    "target_catastrophic": target_c[local].cpu().tolist(),
                    "target_gain": target_gain[local].cpu().tolist(),
                })
    return results


def selected_metrics(predictions, data, threshold):
    selected = []
    for row in predictions:
        branch = int(np.argmax(row["utility"]))
        score = float(row["utility"][branch])
        if score < threshold:
            continue
        index = row["index"]
        selected.append({
            "index": index,
            "sequence": data["sequences"][index],
            "trigger_frame": data["triggers"][index],
            "branch_id": data["branch_names"][index][branch],
            "score": score,
            "label": ("beneficial" if row["target_beneficial"][branch] >= 0.5
                      else ("catastrophic" if
                            row["target_catastrophic"][branch] >= 0.5
                            else "neutral")),
            "target_gain": float(row["target_gain"][branch]),
        })
    counts = {name: sum(row["label"] == name for row in selected)
              for name in ("beneficial", "neutral", "catastrophic")}
    actions = len(selected)
    precision = counts["beneficial"] / actions if actions else 0.0
    return {
        "threshold": float(threshold), "actions": actions,
        "selected_sequences": len({row["sequence"] for row in selected}),
        "precision": precision, **counts, "selected": selected,
    }


def choose_threshold(predictions, data):
    acceptable = []
    for threshold in THRESHOLDS:
        metrics = selected_metrics(predictions, data, threshold)
        if (metrics["actions"] >= 10 and
                metrics["selected_sequences"] >= 4 and
                metrics["precision"] >= 0.95 and
                metrics["catastrophic"] == 0):
            acceptable.append(metrics)
    if not acceptable:
        return None
    acceptable.sort(key=lambda row: (row["actions"],
                                     row["selected_sequences"],
                                     -row["threshold"]), reverse=True)
    result = dict(acceptable[0])
    result.pop("selected")
    return result


def indexes_for_sequences(data, values):
    values = set(values)
    return [index for index, sequence in enumerate(data["sequences"])
            if sequence in values]


def run_smoke(data, seed, output):
    indexes = list(range(len(data["sequences"])))
    train_indexes, test_indexes = indexes[:16], indexes[16:]
    model, trace = train_model(data, train_indexes, seed, epochs=2)
    predictions = predict(model, data, test_indexes)
    result = {
        "schema": "sttrack-lachtt-selector-smoke/v1", "complete": True,
        "scientific_scope": "engineering smoke only; no nested OOF claim",
        "seed": seed, "event_count": len(indexes),
        "parameter_count": parameter_count(model), "train_trace": trace,
        "prediction_count": len(predictions),
        "finite_predictions": all(math.isfinite(value)
            for row in predictions for values in
            (row["utility"], row["beneficial_probability"],
             row["catastrophic_probability"], row["predicted_gain"])
            for value in values),
    }
    output.mkdir(parents=True)
    atomic_json(output / "result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


def main():
    args = parse_args()
    started = time.time()
    args.spec, args.output = args.spec.resolve(), args.output.resolve()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.seed not in MODEL_SEEDS:
        raise ValueError("seed is outside frozen seed set")
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    if spec.get("complete") is not True:
        raise ValueError("selector spec is incomplete")
    if not args.smoke:
        repository = Path(__file__).resolve().parents[1]
        commit = subprocess.check_output(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            text=True).strip()
        if (commit != spec["repository_commit"] or
                sha256_file(Path(__file__).resolve()) !=
                spec["bindings"]["trainer"]["sha256"] or
                args.output != Path(spec["output_root"]).resolve() /
                ("seed%d" % args.seed)):
            raise ValueError("formal selector binding mismatch")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    data_cpu = load_data(spec, smoke=args.smoke)
    data = move_numeric_data(data_cpu, "cuda")
    probe = AssociationSelector().cuda()
    params = parameter_count(probe)
    del probe
    if args.smoke:
        run_smoke(data, args.seed, args.output)
        return

    sequences = sorted(set(data["sequences"]))
    outer_predictions = []
    fold_results = []
    for outer in range(OUTER_FOLDS):
        outer_test_sequences = [sequence for sequence in sequences
            if stable_fold(sequence, "sttrack-lachtt-outer-v1", OUTER_FOLDS) == outer]
        outer_train_sequences = sorted(set(sequences) - set(outer_test_sequences))
        outer_test_indexes = indexes_for_sequences(data, outer_test_sequences)
        outer_train_indexes = indexes_for_sequences(data, outer_train_sequences)
        if not outer_test_indexes or not outer_train_indexes:
            raise RuntimeError("empty outer fold")
        inner_predictions = []
        inner_traces = []
        for inner in range(INNER_FOLDS):
            inner_test_sequences = [sequence for sequence in outer_train_sequences
                if stable_fold(sequence, "sttrack-lachtt-inner-v1-%d" % outer,
                               INNER_FOLDS) == inner]
            inner_train_sequences = sorted(
                set(outer_train_sequences) - set(inner_test_sequences))
            inner_test_indexes = indexes_for_sequences(data, inner_test_sequences)
            inner_train_indexes = indexes_for_sequences(data, inner_train_sequences)
            if not inner_test_indexes or not inner_train_indexes:
                raise RuntimeError("empty inner fold")
            fit_seed = args.seed * 100 + outer * 10 + inner
            model, trace = train_model(data, inner_train_indexes, fit_seed)
            inner_predictions.extend(predict(model, data, inner_test_indexes))
            inner_traces.append(trace)
            del model
            torch.cuda.empty_cache()
        threshold = choose_threshold(inner_predictions, data)
        if threshold is None:
            fold_results.append({
                "outer_fold": outer, "status": "abstain_no_inner_gate",
                "outer_test_sequences": outer_test_sequences,
                "inner_fit_traces": inner_traces,
            })
            continue
        model, outer_trace = train_model(
            data, outer_train_indexes, args.seed * 1000 + outer)
        predictions = predict(model, data, outer_test_indexes)
        metrics = selected_metrics(predictions, data, threshold["threshold"])
        for row in predictions:
            row["outer_fold"] = outer
            row["threshold"] = threshold["threshold"]
        outer_predictions.extend(predictions)
        selected = metrics.pop("selected")
        fold_results.append({
            "outer_fold": outer, "status": "evaluated",
            "outer_test_sequences": outer_test_sequences,
            "inner_threshold": threshold, "outer_metrics": metrics,
            "outer_selected": selected,
            "inner_fit_traces": inner_traces,
            "outer_fit_trace": outer_trace,
        })
        del model
        torch.cuda.empty_cache()
    selected = [row for fold in fold_results
                for row in fold.get("outer_selected", [])]
    counts = {name: sum(row["label"] == name for row in selected)
              for name in ("beneficial", "neutral", "catastrophic")}
    actions = len(selected)
    result = {
        "schema": "sttrack-lachtt-selector-oof-seed/v1",
        "complete": True, "seed": args.seed,
        "event_count": len(data["sequences"]),
        "sequence_count": len(sequences),
        "parameter_count": params,
        "outer_folds": OUTER_FOLDS, "inner_folds": INNER_FOLDS,
        "epochs_per_fit": EPOCHS,
        "fold_results": fold_results,
        "overall": {
            "actions": actions,
            "selected_sequences": len({row["sequence"] for row in selected}),
            "precision": counts["beneficial"] / actions if actions else 0.0,
            **counts,
            "evaluated_outer_folds": sum(
                fold["status"] == "evaluated" for fold in fold_results),
            "abstained_outer_folds": sum(
                fold["status"] != "evaluated" for fold in fold_results),
        },
        "elapsed_seconds": time.time() - started,
        "depthtrack_test_run": False, "cdtb_run": False,
        "vot_low22_run": False, "vot_full127_run": False,
        "automatic_next_stage": False,
    }
    args.output.mkdir(parents=True)
    prediction_path = args.output / "outer_predictions.jsonl.gz"
    atomic_gzip_jsonl(prediction_path, outer_predictions)
    result_path = args.output / "result.json"
    atomic_json(result_path, result)
    manifest = {
        "schema": "sttrack-lachtt-selector-oof-seed-manifest/v1",
        "complete": True, "seed": args.seed,
        "spec": record(args.spec), "trainer": record(Path(__file__).resolve()),
        "result": record(result_path), "predictions": record(prediction_path),
    }
    atomic_json(args.output / "manifest.json", manifest)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
