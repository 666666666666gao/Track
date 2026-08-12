"""Read frame-aligned color descriptions stored as plain text files.

The annotation format is intentionally simple:

    <root>/<sequence>/color_description_ct.txt

For DepthTrack training, each non-empty line describes the RGB frame with the
same zero-based index.  Test-set consumers interpret a one-line file as the
initial-frame description for the whole sequence.  The store itself never
drops blank lines or silently pads data; each consumer enforces its scope.
"""

import os


DESCRIPTION_FILENAME = "color_description_ct.txt"


class ColorDescriptionError(ValueError):
    """Base error for invalid frame-aligned color descriptions."""


class ColorDescriptionAlignmentError(ColorDescriptionError, IndexError):
    """Raised when descriptions cannot be aligned one-to-one with media frames."""


class ColorDescriptionStore(object):
    """Lazily load per-frame plain-text descriptions for one dataset split."""

    def __init__(self, root, filename=DESCRIPTION_FILENAME):
        self.root = os.path.abspath(root) if root else ""
        self.filename = filename
        self._cache = {}

    def path_for(self, sequence_name):
        if not self.root:
            return ""
        return os.path.join(self.root, sequence_name, self.filename)

    def has_sequence(self, sequence_name):
        return os.path.isfile(self.path_for(sequence_name))

    def _load(self, sequence_name):
        if sequence_name in self._cache:
            return self._cache[sequence_name]

        path = self.path_for(sequence_name)
        if not path or not os.path.isfile(path):
            raise ColorDescriptionError(
                "Missing color description file for '{}': {}".format(sequence_name, path))

        with open(path, "r", encoding="utf-8") as handle:
            lines = [line.strip() for line in handle.read().splitlines()]

        if not lines:
            raise ColorDescriptionError("Color description file is empty: {}".format(path))
        blank_indices = [index for index, line in enumerate(lines) if not line]
        if blank_indices:
            raise ColorDescriptionAlignmentError(
                "Color description file contains blank frame descriptions at {}: {}".format(
                    blank_indices[:10], path))

        self._cache[sequence_name] = lines
        return lines

    def frame_count(self, sequence_name):
        return len(self._load(sequence_name))

    def descriptions_for(self, sequence_name, frame_ids):
        lines = self._load(sequence_name)
        descriptions = []
        for frame_id in frame_ids:
            frame_index = int(frame_id)
            if frame_index < 0 or frame_index >= len(lines):
                raise ColorDescriptionAlignmentError(
                    "Description index {} is outside '{}' with {} descriptions".format(
                        frame_index, sequence_name, len(lines)))
            descriptions.append(lines[frame_index])
        return descriptions

    def assert_matches_frame_count(self, sequence_name, frame_count):
        description_count = self.frame_count(sequence_name)
        if description_count != int(frame_count):
            raise ColorDescriptionAlignmentError(
                "Description/frame count mismatch for '{}': {} descriptions, {} frames".format(
                    sequence_name, description_count, frame_count))


def default_color_description_root(project_root, dataset_key):
    """Return the canonical plain-text annotation directory for a dataset."""
    root = os.path.abspath(project_root)
    mapping = {
        "depthtrack_train": ("color_desc", "depthtrack_train"),
        "depthtrack_test": ("color_desc", "depthtrack_test"),
        "cdtb": ("color_desc", "cdtb"),
        "votrgbd2022": ("color_desc", "votrgbd2022"),
    }
    if dataset_key not in mapping:
        raise ValueError("Unsupported color-description dataset: {}".format(dataset_key))
    return os.path.join(root, *mapping[dataset_key])
