"""Build source-specific reports for a combined RGB-D evaluation run."""

from lib.test.evaluation.data import SequenceList


COMBINED_RGBD_DATASETS = frozenset(('rgbd_all', 'rgbd_all_test'))
RGBD_SOURCE_REPORTS = (
    ('depthtrack', 'depthtrack_test'),
    ('cdtb', 'cdtb'),
    ('votrgbd2022', 'votrgbd2022'),
)


def datasets_for_reporting(dataset_name, dataset):
    """Return report names and datasets, splitting a combined RGB-D run by source."""
    normalized_name = dataset_name.lower()
    if normalized_name not in COMBINED_RGBD_DATASETS:
        return [(dataset_name, dataset)]

    sequences = list(dataset)
    expected_sources = {source_name for source_name, _ in RGBD_SOURCE_REPORTS}
    unexpected_sources = sorted({
        sequence.dataset for sequence in sequences
        if sequence.dataset not in expected_sources
    })
    if unexpected_sources:
        raise ValueError(
            'Combined RGB-D results contain unexpected source dataset(s): {}'.format(
                ', '.join(unexpected_sources)))

    reports = []
    for source_name, report_suffix in RGBD_SOURCE_REPORTS:
        source_dataset = SequenceList([
            sequence for sequence in sequences if sequence.dataset == source_name
        ])
        if not source_dataset:
            raise ValueError(
                "Combined RGB-D results are missing source dataset '{}'".format(source_name))
        reports.append((
            '{}__{}'.format(normalized_name, report_suffix),
            source_dataset,
        ))
    return reports
