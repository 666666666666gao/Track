"""Strict loader for sequence-level RGB-D language used at inference."""

import hashlib
import json
import os


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


class RGBDLanguageManifest:
    """Load a clean manifest and fail closed on any provenance mismatch."""

    def __init__(self, path, expected_sha256, expected_dataset,
                 expected_sequence_count):
        self.path = os.path.abspath(os.fspath(path))
        expected_sha256 = str(expected_sha256).strip().lower()
        if len(expected_sha256) != 64:
            raise ValueError('RGB-D language manifest SHA256 must contain 64 hex characters')
        try:
            int(expected_sha256, 16)
        except ValueError as error:
            raise ValueError('RGB-D language manifest SHA256 is not hexadecimal') from error
        observed_sha256 = sha256_file(self.path)
        if observed_sha256 != expected_sha256:
            raise ValueError(
                'RGB-D language manifest SHA256 mismatch: expected {}, observed {}'.format(
                    expected_sha256, observed_sha256))

        if isinstance(expected_sequence_count, bool):
            raise ValueError('Expected RGB-D language sequence count must be a positive integer')
        expected_sequence_count = int(expected_sequence_count)
        if expected_sequence_count <= 0:
            raise ValueError('Expected RGB-D language sequence count must be a positive integer')
        expected_dataset = str(expected_dataset).strip().lower()
        if not expected_dataset:
            raise ValueError('Expected RGB-D language dataset must be non-empty')

        records = {}
        with open(self.path, 'r', encoding='utf-8') as stream:
            for line_number, raw_line in enumerate(stream, start=1):
                if not raw_line.strip():
                    raise ValueError('Blank RGB-D language row at line {}'.format(line_number))
                record = json.loads(raw_line)
                if not isinstance(record, dict):
                    raise ValueError('RGB-D language row {} is not an object'.format(line_number))
                dataset = str(record.get('dataset', '')).strip().lower()
                sequence_name = str(record.get('sequence_name', '')).strip()
                language = str(record.get('language', '')).strip()
                quality = record.get('annotation_quality')
                if dataset != expected_dataset:
                    raise ValueError('Unexpected dataset at RGB-D language row {}'.format(line_number))
                if not sequence_name or not language:
                    raise ValueError('Missing sequence name or language at row {}'.format(line_number))
                if sequence_name in records:
                    raise ValueError('Duplicate RGB-D language sequence {}'.format(sequence_name))
                if (not isinstance(quality, dict) or
                        quality.get('is_valid') is not True or
                        quality.get('has_bbox_leak') is not False or
                        quality.get('has_absolute_path') is not False):
                    raise ValueError('Unsafe RGB-D language annotation at row {}'.format(line_number))
                records[sequence_name] = language

        if len(records) != expected_sequence_count:
            raise ValueError(
                'RGB-D language sequence count mismatch: expected {}, observed {}'.format(
                    expected_sequence_count, len(records)))
        self.sha256 = observed_sha256
        self.dataset = expected_dataset
        self.records = records

    def language_for(self, sequence_name):
        sequence_name = str(sequence_name).strip()
        try:
            return self.records[sequence_name]
        except KeyError as error:
            raise KeyError(
                'Sequence {!r} is absent from the bound RGB-D language manifest'.format(
                    sequence_name)) from error


class RGBDAnchorLanguageManifest:
    """Load identity-only text keyed by a VOT sequence and anchor frame.

    Every record is tied to the frame used to initialize one multi-start
    trajectory.  The loader intentionally accepts no sequence-level fallback:
    a missing anchor is an experiment-integrity error, not permission to reuse
    stale frame-zero state language.
    """

    def __init__(self, path, expected_sha256, expected_dataset,
                 expected_record_count):
        self.path = os.path.abspath(os.fspath(path))
        expected_sha256 = str(expected_sha256).strip().lower()
        if len(expected_sha256) != 64:
            raise ValueError(
                'RGB-D anchor language manifest SHA256 must contain 64 hex characters')
        try:
            int(expected_sha256, 16)
        except ValueError as error:
            raise ValueError(
                'RGB-D anchor language manifest SHA256 is not hexadecimal') from error
        observed_sha256 = sha256_file(self.path)
        if observed_sha256 != expected_sha256:
            raise ValueError(
                'RGB-D anchor language manifest SHA256 mismatch: expected {}, observed {}'.format(
                    expected_sha256, observed_sha256))

        if isinstance(expected_record_count, bool):
            raise ValueError('Expected RGB-D anchor record count must be positive')
        expected_record_count = int(expected_record_count)
        if expected_record_count <= 0:
            raise ValueError('Expected RGB-D anchor record count must be positive')
        expected_dataset = str(expected_dataset).strip().lower()
        if not expected_dataset:
            raise ValueError('Expected RGB-D language dataset must be non-empty')

        records = {}
        with open(self.path, 'r', encoding='utf-8') as stream:
            for line_number, raw_line in enumerate(stream, start=1):
                if not raw_line.strip():
                    raise ValueError(
                        'Blank RGB-D anchor language row at line {}'.format(line_number))
                record = json.loads(raw_line)
                if not isinstance(record, dict):
                    raise ValueError(
                        'RGB-D anchor language row {} is not an object'.format(line_number))
                dataset = str(record.get('dataset', '')).strip().lower()
                sequence_name = str(record.get('sequence_name', '')).strip()
                anchor_index = record.get('anchor_index')
                language = str(record.get('language', '')).strip()
                quality = record.get('annotation_quality')
                if dataset != expected_dataset:
                    raise ValueError(
                        'Unexpected dataset at RGB-D anchor language row {}'.format(
                            line_number))
                if (not sequence_name or isinstance(anchor_index, bool) or
                        not isinstance(anchor_index, int) or anchor_index < 0 or
                        not language):
                    raise ValueError(
                        'Missing sequence, anchor, or language at row {}'.format(
                            line_number))
                key = (sequence_name, anchor_index)
                if key in records:
                    raise ValueError(
                        'Duplicate RGB-D anchor language record {}@{}'.format(*key))
                if (not isinstance(quality, dict) or
                        quality.get('is_valid') is not True or
                        quality.get('identity_only') is not True or
                        quality.get('uses_future_frame') is not False or
                        quality.get('has_bbox_leak') is not False or
                        quality.get('has_absolute_path') is not False):
                    raise ValueError(
                        'Unsafe RGB-D anchor annotation at row {}'.format(line_number))
                records[key] = language

        if len(records) != expected_record_count:
            raise ValueError(
                'RGB-D anchor language record count mismatch: expected {}, observed {}'.format(
                    expected_record_count, len(records)))
        self.sha256 = observed_sha256
        self.dataset = expected_dataset
        self.records = records

    def language_for(self, sequence_name, anchor_index):
        sequence_name = str(sequence_name).strip()
        if isinstance(anchor_index, bool):
            raise ValueError('VOT anchor index must be a non-negative integer')
        anchor_index = int(anchor_index)
        key = (sequence_name, anchor_index)
        try:
            return self.records[key]
        except KeyError as error:
            raise KeyError(
                'Anchor {!r}@{} is absent from the bound RGB-D language manifest'.format(
                    sequence_name, anchor_index)) from error


__all__ = [
    'RGBDLanguageManifest', 'RGBDAnchorLanguageManifest', 'sha256_file']
