"""CPU tests for anchor identity and directional trajectory coverage."""
from copy import deepcopy
from io import StringIO
from types import SimpleNamespace
import unittest

from vot_trajectory_coverage import checked_length, declared_anchors, read_properties


class CoverageTests(unittest.TestCase):
    def test_official_property_names_and_raw_stream_lengths(self):
        streams = {"seq_00000020_confidence.value": "\n0.8\n",
                   "seq_00000020_time.value": "0.01\n0.02\n"}
        results = SimpleNamespace(read=lambda name: StringIO(streams[name]))
        trajectory = SimpleNamespace(properties=lambda: ("confidence", "time"))
        properties = read_properties(results, "seq_00000020", trajectory)
        self.assertEqual({key: len(value) for key, value in properties.items()},
                         {"confidence": 2, "time": 2})
        row = dict(index=20, direction="forward", estimated_frames=2, name="seq_00000020")
        self.assertEqual(checked_length(row, 22, 2, properties), 2)
        streams["seq_00000020_time.value"] = "0.01\n"
        with self.assertRaises(AssertionError):
            checked_length(row, 22, 2, read_properties(results, row["name"], trajectory))

    def manifest(self):
        names = ["seq%03d" % i for i in range(127)]
        shards, offset = [], 0
        for index, count in enumerate((441, 441, 441, 442)):
            anchors = [dict(sequence=names[i % 127], index=i, direction="forward",
                            estimated_frames=2000 - i) for i in range(offset, offset + count)]
            shards.append(dict(index=index, anchor_count=count, anchors=anchors,
                expected_trajectories=["%s_%08d" % (row["sequence"], row["index"]) for row in anchors]))
            offset += count
        return dict(schema="M122_full127_frozen_shards_v1", sequences=names,
            source={name: {} for name in names}, total_anchor_count=1765, shards=shards)

    def test_directional_lengths_and_auxiliary_streams(self):
        row = dict(index=20, direction="forward", estimated_frames=80, name="seq_00000020")
        props = dict(confidence=[None] * 80, time=[0.] * 80)
        self.assertEqual(checked_length(row, 100, 80, props), 80)
        reverse = dict(row, direction="backward", estimated_frames=21)
        self.assertEqual(checked_length(reverse, 100, 21,
            dict(confidence=[None] * 21, time=[0.] * 21)), 21)
        with self.assertRaises(AssertionError):
            checked_length(row, 100, 21, props)
        with self.assertRaises(AssertionError):
            checked_length(row, 100, 80, dict(props, confidence=[None] * 79))

    def test_exact_anchor_partition(self):
        manifest = self.manifest()
        self.assertEqual(len(declared_anchors(manifest)), 1765)
        changed = deepcopy(manifest)
        changed["shards"][1]["anchors"][0] = changed["shards"][0]["anchors"][0]
        with self.assertRaises(AssertionError):
            declared_anchors(changed)

    def test_wrong_name_or_sequence_is_rejected(self):
        manifest = self.manifest()
        manifest["shards"][0]["expected_trajectories"][0] = "wrong_00000000"
        with self.assertRaises(AssertionError):
            declared_anchors(manifest)
        manifest = self.manifest()
        manifest["shards"][0]["anchors"][0]["sequence"] = "unknown"
        with self.assertRaises(AssertionError):
            declared_anchors(manifest)


if __name__ == "__main__":
    unittest.main()
