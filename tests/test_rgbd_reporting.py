import types
import unittest


class RGBDReportingTest(unittest.TestCase):
    def test_combined_results_are_reported_once_per_source_dataset(self):
        from lib.test.analysis.rgbd_reports import datasets_for_reporting

        sequences = [
            types.SimpleNamespace(name="depthtrack__adapter01_indoor", dataset="depthtrack"),
            types.SimpleNamespace(name="cdtb__backpack_blue", dataset="cdtb"),
            types.SimpleNamespace(name="votrgbd2022__cartman_1", dataset="votrgbd2022"),
        ]

        reports = datasets_for_reporting("rgbd_all_test", sequences)

        self.assertEqual(
            [report_name for report_name, _ in reports],
            [
                "rgbd_all_test__depthtrack_test",
                "rgbd_all_test__cdtb",
                "rgbd_all_test__votrgbd2022",
            ],
        )
        self.assertEqual(
            [[sequence.name for sequence in dataset] for _, dataset in reports],
            [
                ["depthtrack__adapter01_indoor"],
                ["cdtb__backpack_blue"],
                ["votrgbd2022__cartman_1"],
            ],
        )

    def test_regular_dataset_keeps_the_existing_single_report(self):
        from lib.test.analysis.rgbd_reports import datasets_for_reporting

        sequences = [types.SimpleNamespace(name="adapter01_indoor", dataset="depthtrack")]

        reports = datasets_for_reporting("depthtrack_test", sequences)

        self.assertEqual(reports, [("depthtrack_test", sequences)])

    def test_combined_report_fails_if_an_expected_source_is_missing(self):
        from lib.test.analysis.rgbd_reports import datasets_for_reporting

        sequences = [types.SimpleNamespace(name="depthtrack__adapter01_indoor", dataset="depthtrack")]

        with self.assertRaisesRegex(ValueError, "missing source dataset"):
            datasets_for_reporting("rgbd_all", sequences)

    def test_combined_report_rejects_an_unknown_source_dataset(self):
        from lib.test.analysis.rgbd_reports import datasets_for_reporting

        sequences = [
            types.SimpleNamespace(name="depthtrack__adapter01_indoor", dataset="depthtrack"),
            types.SimpleNamespace(name="cdtb__backpack_blue", dataset="cdtb"),
            types.SimpleNamespace(name="votrgbd2022__cartman_1", dataset="votrgbd2022"),
            types.SimpleNamespace(name="typo__unknown", dataset="votrgbd2202"),
        ]

        with self.assertRaisesRegex(ValueError, "unexpected source dataset"):
            datasets_for_reporting("rgbd_all_test", sequences)


if __name__ == "__main__":
    unittest.main()
