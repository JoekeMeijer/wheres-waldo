import pytest

from src.eval.metrics import DetectionMatch, match_detections, mean_localization_error, precision_recall


def test_match_detections_counts_duplicates_as_false_positives():
    result = match_detections(
        pred_boxes=[(0, 0, 10, 10), (1, 1, 11, 11), (30, 30, 40, 40)],
        gt_boxes=[(0, 0, 10, 10), (50, 50, 60, 60)],
    )

    assert result.true_positives == 1
    assert result.false_positives == 2
    assert result.false_negatives == 1
    assert result.localization_errors == [0.0]


def test_precision_recall_and_localization_error():
    matches = [
        DetectionMatch(2, 1, 0, [3.0, 5.0]),
        DetectionMatch(1, 0, 1, [7.0]),
    ]

    assert precision_recall(matches) == pytest.approx((3 / 4, 3 / 4))
    assert mean_localization_error(matches) == pytest.approx(5.0)


def test_empty_metrics_are_explicit():
    assert precision_recall([]) == (0.0, 0.0)
    assert mean_localization_error([]) is None
