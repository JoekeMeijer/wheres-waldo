import numpy as np
import pytest

from src.eval.box_utils import box_area, iou, iou_matrix, non_max_suppression


@pytest.mark.parametrize(
    ("box", "expected"),
    [
        ((1, 2, 11, 7), 50.0),
        ((1, 2, 1, 7), 0.0),
        ((1, 2, 11, 2), 0.0),
        ((11, 7, 1, 2), 0.0),
    ],
)
def test_box_area_clamps_degenerate_boxes(box, expected):
    assert box_area(np.array(box)) == expected


def test_iou_handles_common_overlap_cases():
    assert iou(np.array((0, 0, 10, 10)), np.array((0, 0, 10, 10))) == 1.0
    assert iou(np.array((0, 0, 10, 10)), np.array((20, 20, 30, 30))) == 0.0
    assert iou(np.array((0, 0, 10, 10)), np.array((5, 5, 15, 15))) == pytest.approx(25 / 175)
    assert iou(np.array((0, 0, 10, 10)), np.array((2, 2, 8, 8))) == pytest.approx(36 / 100)


def test_iou_matrix_matches_scalar_iou():
    boxes_a = np.array(((0, 0, 10, 10), (20, 20, 30, 30)))
    boxes_b = np.array(((5, 5, 15, 15), (0, 0, 10, 10)))

    expected = np.array([[iou(a, b) for b in boxes_b] for a in boxes_a])
    np.testing.assert_allclose(iou_matrix(boxes_a, boxes_b), expected)


def test_non_max_suppression_keeps_highest_scoring_overlapping_box():
    boxes = np.array(((0, 0, 10, 10), (1, 1, 11, 11), (30, 30, 40, 40)))
    scores = np.array((0.8, 0.95, 0.7))

    assert non_max_suppression(boxes, scores, iou_threshold=0.3) == [1, 2]
