#!/usr/bin/env python3
"""Train a frozen-backbone SUTrack rollback gate with sequence-group OOF."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import tempfile

import numpy as np
import torch


SCHEMA = 'sutrack-state-gate-training/v1'
ARTIFACT_SCHEMA = 'sutrack-state-gate-artifact/v1'
OOF_SCHEMA = 'sutrack-state-gate-oof-row/v1'
TEMPORAL_FEATURE_SCHEMA = 'current-delta1-mean2/v1'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--capacity-result', type=Path, required=True)
    parser.add_argument('--split-plan', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--learning-rate', type=float, default=0.03)
    parser.add_argument('--weight-decay', type=float, default=1.0e-3)
    parser.add_argument('--maximum-positive-weight', type=float, default=20.0)
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + '.', suffix='.tmp', dir=str(path.parent))
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_torch_save(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + '.', suffix='.tmp', dir=str(path.parent))
    os.close(descriptor)
    try:
        torch.save(payload, temporary)
        with open(temporary, 'rb') as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_json(path):
    with Path(path).open('r', encoding='utf-8') as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError('{} is not an object'.format(path))
    return value


def load_jsonl(path):
    records = []
    with Path(path).open('r', encoding='utf-8') as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            if not raw_line.strip():
                raise ValueError('blank row {}:{}'.format(path, line_number))
            record = json.loads(raw_line)
            if not isinstance(record, dict):
                raise ValueError('non-object row {}:{}'.format(path, line_number))
            records.append(record)
    return records


def temporalize(records, base_feature_names):
    by_sequence = defaultdict(list)
    for record in records:
        by_sequence[record['sequence']].append(record)
    output = {}
    for sequence, sequence_rows in by_sequence.items():
        sequence_rows.sort(key=lambda row: int(row['frame_index']))
        history = []
        previous_index = None
        for row in sequence_rows:
            frame_index = int(row['frame_index'])
            if previous_index is not None and frame_index != previous_index + 1:
                raise ValueError('non-contiguous trace for {}'.format(sequence))
            current = np.asarray(row['features'], dtype=np.float64)
            if (current.shape != (len(base_feature_names),) or
                    not np.isfinite(current).all() or
                    row.get('feature_names') != base_feature_names):
                raise ValueError('feature contract mismatch {}:{}'.format(
                    sequence, frame_index))
            previous = history[-1] if history else current
            mean_two = (np.mean(history[-2:], axis=0)
                        if history else current)
            vector = np.concatenate((current, current - previous, mean_two))
            if not np.isfinite(vector).all():
                raise ValueError('non-finite temporal feature')
            key = (sequence, frame_index)
            if key in output:
                raise ValueError('duplicate temporal row {}'.format(key))
            output[key] = vector
            history.append(current)
            previous_index = frame_index
    names = (["current__" + name for name in base_feature_names] +
             ["delta1__" + name for name in base_feature_names] +
             ["mean2__" + name for name in base_feature_names])
    return output, names


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def fit_linear(rows, vectors, seed, epochs, learning_rate, weight_decay,
               maximum_positive_weight):
    if not rows:
        raise ValueError('empty training fold')
    set_seed(seed)
    keys = [(row['sequence'], int(row['frame_index'])) for row in rows]
    x = np.stack([vectors[key] for key in keys])
    y = np.asarray([bool(row['rollback_beneficial']) for row in rows],
                   dtype=np.float64)
    positives = int(y.sum())
    negatives = len(y) - positives
    if positives <= 0 or negatives <= 0:
        raise ValueError('training fold lacks both classes')
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std < 1.0e-6] = 1.0
    normalized = (x - mean) / std
    sequence_counts = Counter(row['sequence'] for row in rows)
    sequence_weight = np.asarray(
        [1.0 / sequence_counts[row['sequence']] for row in rows],
        dtype=np.float64)
    sequence_weight *= len(sequence_weight) / sequence_weight.sum()
    positive_weight = min(
        float(maximum_positive_weight), float(negatives / positives))
    class_weight = np.where(y > 0.5, positive_weight, 1.0)
    sample_weight = sequence_weight * class_weight
    sample_weight /= sample_weight.mean()

    x_tensor = torch.from_numpy(normalized)
    y_tensor = torch.from_numpy(y).reshape(-1, 1)
    weight_tensor = torch.from_numpy(sample_weight).reshape(-1, 1)
    model = torch.nn.Linear(x.shape[1], 1, bias=True).double()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(learning_rate),
        weight_decay=float(weight_decay))
    for _ in range(int(epochs)):
        optimizer.zero_grad(set_to_none=True)
        logits = model(x_tensor)
        loss_rows = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, y_tensor, reduction='none')
        loss = (loss_rows * weight_tensor).mean()
        if not bool(torch.isfinite(loss)):
            raise ValueError('non-finite training loss')
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        final_loss = float((
            torch.nn.functional.binary_cross_entropy_with_logits(
                model(x_tensor), y_tensor, reduction='none') *
            weight_tensor).mean().item())
    return {
        'weight': model.weight.detach().cpu().numpy().reshape(-1),
        'bias': float(model.bias.detach().cpu().item()),
        'mean': mean,
        'std': std,
        'positive_weight': positive_weight,
        'training_rows': len(rows),
        'training_positive_rows': positives,
        'training_loss': final_loss,
    }


def predict(model, rows, vectors):
    if not rows:
        return np.empty((0,), dtype=np.float64)
    x = np.stack([
        vectors[(row['sequence'], int(row['frame_index']))]
        for row in rows])
    normalized = (x - model['mean']) / model['std']
    logits = normalized.dot(model['weight']) + model['bias']
    probabilities = np.empty_like(logits)
    positive = logits >= 0.0
    probabilities[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    exp_logits = np.exp(logits[~positive])
    probabilities[~positive] = exp_logits / (1.0 + exp_logits)
    if not np.isfinite(probabilities).all():
        raise ValueError('non-finite model probability')
    return probabilities


def action_metrics(rows, probabilities, threshold):
    selected = [
        (row, float(probability))
        for row, probability in zip(rows, probabilities)
        if float(probability) >= float(threshold)]
    actions = len(selected)
    positives = sum(
        int(bool(row['rollback_beneficial'])) for row, _ in selected)
    harmful = sum(int(bool(row['rollback_harmful'])) for row, _ in selected)
    catastrophic = sum(
        int(bool(row['rollback_catastrophic_harm']))
        for row, _ in selected)
    net_gain = sum(float(row['rollback_delta_iou']) for row, _ in selected)
    return {
        'threshold': float(threshold),
        'action_rows': actions,
        'beneficial_action_rows': positives,
        'harmful_action_rows': harmful,
        'catastrophic_harm_rows': catastrophic,
        'precision': (float(positives / actions) if actions else 0.0),
        'harm_rate': (float(harmful / actions) if actions else 0.0),
        'net_iou_gain': float(net_gain),
        'action_sequences': len(set(
            row['sequence'] for row, _ in selected)),
    }


def passes(metrics, gate, audit=False):
    minimum_gain = (gate['minimum_net_iou_gain_exclusive']
                    if audit else gate['minimum_net_iou_gain'])
    gain_pass = (metrics['net_iou_gain'] > minimum_gain
                 if audit else metrics['net_iou_gain'] >= minimum_gain)
    return bool(
        metrics['precision'] >= gate['minimum_precision'] and
        metrics['harm_rate'] <= gate['maximum_harm_rate'] and
        metrics['catastrophic_harm_rows'] <=
        gate['maximum_catastrophic_harm_rows'] and
        metrics['action_rows'] >= gate['minimum_action_rows'] and
        metrics['action_sequences'] >= gate['minimum_action_sequences'] and
        gain_pass)


def select_threshold(rows, probabilities, gate):
    if len(rows) != len(probabilities) or not rows:
        raise ValueError('threshold input mismatch')
    order = sorted(
        range(len(rows)),
        key=lambda index: (-float(probabilities[index]),
                           rows[index]['sequence'],
                           int(rows[index]['frame_index'])))
    sorted_rows = [rows[index] for index in order]
    sorted_probabilities = np.asarray(
        [probabilities[index] for index in order], dtype=np.float64)
    cumulative_positive = 0
    cumulative_harmful = 0
    cumulative_catastrophic = 0
    cumulative_gain = 0.0
    action_sequences = set()
    candidates = []
    for index, row in enumerate(sorted_rows):
        cumulative_positive += int(bool(row['rollback_beneficial']))
        cumulative_harmful += int(bool(row['rollback_harmful']))
        cumulative_catastrophic += int(
            bool(row['rollback_catastrophic_harm']))
        cumulative_gain += float(row['rollback_delta_iou'])
        action_sequences.add(row['sequence'])
        next_probability = (sorted_probabilities[index + 1]
                            if index + 1 < len(sorted_rows) else -1.0)
        if sorted_probabilities[index] == next_probability:
            continue
        actions = index + 1
        metrics = {
            'threshold': float(sorted_probabilities[index]),
            'action_rows': actions,
            'beneficial_action_rows': cumulative_positive,
            'harmful_action_rows': cumulative_harmful,
            'catastrophic_harm_rows': cumulative_catastrophic,
            'precision': float(cumulative_positive / actions),
            'harm_rate': float(cumulative_harmful / actions),
            'net_iou_gain': float(cumulative_gain),
            'action_sequences': len(action_sequences),
        }
        if passes(metrics, gate, audit=False):
            candidates.append(metrics)
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda item: (item['net_iou_gain'],
                          item['beneficial_action_rows'],
                          item['precision'], item['threshold']))


def json_model_summary(model):
    return {
        'input_features': int(len(model['weight'])),
        'parameters': int(len(model['weight']) + 1),
        'positive_weight': float(model['positive_weight']),
        'training_rows': int(model['training_rows']),
        'training_positive_rows': int(model['training_positive_rows']),
        'training_loss': float(model['training_loss']),
    }


def main():
    args = parse_args()
    if (args.epochs <= 0 or not math.isfinite(args.learning_rate) or
            args.learning_rate <= 0.0 or
            not math.isfinite(args.weight_decay) or args.weight_decay < 0.0 or
            not math.isfinite(args.maximum_positive_weight) or
            args.maximum_positive_weight < 1.0):
        raise ValueError('invalid optimization settings')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError('refusing non-empty output {}'.format(output_root))
    output_root.mkdir(parents=True, exist_ok=True)

    result_path = args.capacity_result.resolve()
    split_path = args.split_plan.resolve()
    capacity = load_json(result_path)
    split = load_json(split_path)
    if (capacity.get('complete') is not True or
            capacity.get('capacity_supported') is not True or
            capacity.get('public_evaluation') is not False or
            capacity.get('ground_truth_join') != 'strictly_post_inference'):
        raise ValueError('capacity result is not eligible')
    if (split.get('schema') != 'sutrack-state-gate-split-plan/v1' or
            split.get('complete') is not True or
            split.get('created_before_full152_gt_join') is not True or
            split.get('outcome_fields_used_for_split') != [] or
            split.get('public_evaluation') is not False or
            split.get('audit_consumption_limit') != 1):
        raise ValueError('split plan contract failed')
    rows_path = Path(capacity['rows_path']).resolve()
    if sha256_file(rows_path) != capacity.get('rows_sha256'):
        raise ValueError('capacity rows SHA mismatch')
    records = load_jsonl(rows_path)
    if len(records) != capacity.get('row_count'):
        raise ValueError('capacity row count mismatch')
    expected_sequences = list(capacity['expected_sequences'])
    split_sequences = (
        list(split['calibration_sequences']) + list(split['audit_sequences']))
    if (len(expected_sequences) != 152 or
            set(expected_sequences) != set(split_sequences) or
            len(set(split_sequences)) != 152):
        raise ValueError('Train152 split/row scope mismatch')
    base_feature_names = list(capacity['feature_names'])
    vectors, temporal_feature_names = temporalize(records, base_feature_names)
    candidates = [
        row for row in records
        if row.get('label_available') is True and
        row.get('hard_conflict') is True]
    calibration_set = set(split['calibration_sequences'])
    audit_set = set(split['audit_sequences'])
    calibration_rows = [
        row for row in candidates if row['sequence'] in calibration_set]
    audit_rows = [row for row in candidates if row['sequence'] in audit_set]
    if (not calibration_rows or not audit_rows or
            any(row['sequence'] in audit_set for row in calibration_rows) or
            any(row['sequence'] in calibration_set for row in audit_rows)):
        raise ValueError('candidate role split failed')

    threshold_gate = split['threshold_selection']
    seed_reports = []
    trained_models = {}
    artifact_records = []
    for seed in split['training_seeds']:
        oof_rows = []
        oof_probabilities = []
        seen_keys = set()
        fold_summaries = []
        for fold_record in split['folds']:
            fold = int(fold_record['fold'])
            validation_sequences = set(fold_record['sequences'])
            train_rows = [
                row for row in calibration_rows
                if row['sequence'] not in validation_sequences]
            validation_rows = [
                row for row in calibration_rows
                if row['sequence'] in validation_sequences]
            model = fit_linear(
                train_rows, vectors, int(seed) + 1000 * (fold + 1),
                args.epochs, args.learning_rate, args.weight_decay,
                args.maximum_positive_weight)
            probabilities = predict(model, validation_rows, vectors)
            for row, probability in zip(validation_rows, probabilities):
                key = (row['sequence'], int(row['frame_index']))
                if key in seen_keys:
                    raise ValueError('OOF row predicted twice {}'.format(key))
                seen_keys.add(key)
                oof_rows.append(row)
                oof_probabilities.append(float(probability))
            fold_summaries.append({
                'fold': fold,
                'training_sequences': len(set(
                    row['sequence'] for row in train_rows)),
                'validation_sequences': len(validation_sequences),
                'training_rows': len(train_rows),
                'validation_rows': len(validation_rows),
                'model': json_model_summary(model),
            })
        expected_keys = set(
            (row['sequence'], int(row['frame_index']))
            for row in calibration_rows)
        if seen_keys != expected_keys:
            raise ValueError('OOF coverage is incomplete')
        selection = select_threshold(
            oof_rows, np.asarray(oof_probabilities), threshold_gate)
        seed_pass = selection is not None
        final_model = fit_linear(
            calibration_rows, vectors, int(seed), args.epochs,
            args.learning_rate, args.weight_decay,
            args.maximum_positive_weight)
        trained_models[int(seed)] = final_model
        oof_path = output_root / 'seed_{}'.format(seed) / 'oof.jsonl'
        oof_payload = []
        threshold = (float(selection['threshold']) if selection is not None
                     else 1.0)
        for row, probability in sorted(
                zip(oof_rows, oof_probabilities),
                key=lambda item: (item[0]['sequence'],
                                  int(item[0]['frame_index']))):
            oof_payload.append({
                'schema': OOF_SCHEMA,
                'seed': int(seed),
                'sequence': row['sequence'],
                'frame_index': int(row['frame_index']),
                'probability': float(probability),
                'action': bool(float(probability) >= threshold),
                'rollback_beneficial': bool(row['rollback_beneficial']),
                'rollback_harmful': bool(row['rollback_harmful']),
                'rollback_catastrophic_harm': bool(
                    row['rollback_catastrophic_harm']),
                'rollback_delta_iou': float(row['rollback_delta_iou']),
                'role': 'calibration_sequence_group_oof',
            })
        atomic_write(oof_path, ''.join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n'
            for row in oof_payload).encode('utf-8'))
        artifact_path = output_root / 'seed_{}'.format(seed) / 'artifact.pt'
        artifact = {
            'schema': ARTIFACT_SCHEMA,
            'seed': int(seed),
            'eligible_from_oof': bool(seed_pass),
            'threshold': threshold,
            'weight': torch.from_numpy(final_model['weight'].copy()),
            'bias': float(final_model['bias']),
            'mean': torch.from_numpy(final_model['mean'].copy()),
            'std': torch.from_numpy(final_model['std'].copy()),
            'base_feature_names': base_feature_names,
            'feature_names': temporal_feature_names,
            'temporal_feature_schema': TEMPORAL_FEATURE_SCHEMA,
            'model_family': split['model_family'],
            'backbone_frozen': True,
            'maximum_consecutive_gate_rollbacks': split[
                'recursive_audit_gate'][
                    'maximum_consecutive_gate_rollbacks'],
            'cooldown_frames_after_rollback': split[
                'recursive_audit_gate']['cooldown_frames_after_rollback'],
            'capacity_result_path': str(result_path),
            'capacity_result_sha256': sha256_file(result_path),
            'capacity_rows_path': str(rows_path),
            'capacity_rows_sha256': sha256_file(rows_path),
            'split_plan_path': str(split_path),
            'split_plan_sha256': sha256_file(split_path),
            'oof_path': str(oof_path),
            'oof_sha256': sha256_file(oof_path),
            'oof_metrics': selection,
            'calibration_sequences': list(split['calibration_sequences']),
            'audit_sequences': list(split['audit_sequences']),
            'audit_evaluated': False,
            'future_frame_text_used': False,
            'public_evaluation': False,
            'source_trace_contract': capacity['trace_contract'],
        }
        atomic_torch_save(artifact_path, artifact)
        artifact_records.append({
            'seed': int(seed), 'path': str(artifact_path),
            'sha256': sha256_file(artifact_path),
            'bytes': artifact_path.stat().st_size,
        })
        seed_reports.append({
            'seed': int(seed),
            'oof_passed': bool(seed_pass),
            'oof_selection': selection,
            'oof_path': str(oof_path),
            'oof_sha256': sha256_file(oof_path),
            'folds': fold_summaries,
            'final_model': json_model_summary(final_model),
        })

    all_seeds_passed = all(report['oof_passed'] for report in seed_reports)
    deployment_seed = int(split['deployment_seed'])
    if deployment_seed not in trained_models:
        raise ValueError('deployment seed was not trained')
    deployment_report = next(
        report for report in seed_reports
        if report['seed'] == deployment_seed)
    audit_evaluated = False
    audit_metrics = None
    immediate_audit_passed = False
    if all_seeds_passed:
        # This is the one and only audit-label policy evaluation.  No seed or
        # threshold is selected with these observations.
        audit_evaluated = True
        audit_probabilities = predict(
            trained_models[deployment_seed], audit_rows, vectors)
        audit_metrics = action_metrics(
            audit_rows, audit_probabilities,
            deployment_report['oof_selection']['threshold'])
        immediate_audit_passed = passes(
            audit_metrics, split['immediate_audit_gate'], audit=True)

    ready_for_recursive_audit = bool(
        all_seeds_passed and immediate_audit_passed)
    result = {
        'schema': SCHEMA,
        'complete': True,
        'decision': ('ready_for_recursive_audit'
                     if ready_for_recursive_audit else
                     'training_rejected'),
        'ready_for_recursive_audit': ready_for_recursive_audit,
        'all_seeds_oof_passed': all_seeds_passed,
        'immediate_audit_evaluated': audit_evaluated,
        'immediate_audit_policies_evaluated': int(audit_evaluated),
        'immediate_audit_passed': immediate_audit_passed,
        'immediate_audit_metrics': audit_metrics,
        'deployment_seed': deployment_seed,
        'seed_selection_used_audit': False,
        'audit_consumption_limit': split['audit_consumption_limit'],
        'calibration_candidate_rows': len(calibration_rows),
        'audit_candidate_rows': len(audit_rows),
        'base_feature_names': base_feature_names,
        'feature_names': temporal_feature_names,
        'temporal_feature_schema': TEMPORAL_FEATURE_SCHEMA,
        'optimization': {
            'epochs': args.epochs,
            'learning_rate': args.learning_rate,
            'weight_decay': args.weight_decay,
            'maximum_positive_weight': args.maximum_positive_weight,
        },
        'seed_reports': seed_reports,
        'artifacts': artifact_records,
        'capacity_result_path': str(result_path),
        'capacity_result_sha256': sha256_file(result_path),
        'capacity_rows_path': str(rows_path),
        'capacity_rows_sha256': sha256_file(rows_path),
        'split_plan_path': str(split_path),
        'split_plan_sha256': sha256_file(split_path),
        'trainer_path': str(Path(__file__).resolve()),
        'trainer_sha256': sha256_file(Path(__file__).resolve()),
        'backbone_frozen': True,
        'future_frame_text_used': False,
        'public_evaluation': False,
    }
    # Recheck every frozen input before publishing the terminal decision.
    if (sha256_file(result_path) != result['capacity_result_sha256'] or
            sha256_file(rows_path) != result['capacity_rows_sha256'] or
            sha256_file(split_path) != result['split_plan_sha256']):
        raise ValueError('source drift during training')
    output_path = output_root / 'training_result.json'
    atomic_write(
        output_path,
        (json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2,
                    allow_nan=False) + '\n').encode('utf-8'))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2,
                     allow_nan=False))


if __name__ == '__main__':
    main()
