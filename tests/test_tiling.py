from types import SimpleNamespace

import numpy as np
from PIL import Image

from src.data.tiling import _tile_labels, _visible_fraction, predict_multiscale, predict_tiled, tile_grid


def test_tile_grid_covers_small_and_large_images_in_bounds():
    assert tile_grid(400, 300, tile=640, overlap=100) == [(0, 0, 640, 640)]

    tiles = tile_grid(1000, 900, tile=640, overlap=240)
    assert tiles
    assert all(0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 900 for x1, y1, x2, y2 in tiles)
    assert any(x1 == 0 and y1 == 0 for x1, y1, _, _ in tiles)
    assert any(x2 == 1000 and y2 == 900 for _, _, x2, y2 in tiles)


def test_tile_labels_translates_complete_boxes_and_rejects_partial_boxes():
    tile = (10, 20, 110, 120)
    assert _visible_fraction((20, 30, 50, 60), tile) == 1.0
    assert _tile_labels([(20, 30, 50, 60)], tile) == [(10, 10, 40, 40)]
    assert _tile_labels([(0, 30, 50, 60)], tile) is None
    assert _tile_labels([(200, 200, 220, 220)], tile) == []


class _FakeBox:
    def __init__(self, coordinates, confidence):
        self.xyxy = [np.asarray(coordinates)]
        self.conf = SimpleNamespace(item=lambda: confidence)


class _FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes


class _FakeModel:
    def __init__(self, detections_by_call):
        self.detections_by_call = iter(detections_by_call)
        self.calls = []

    def predict(self, images, **kwargs):
        self.calls.append((images, kwargs))
        return next(self.detections_by_call)


def test_predict_tiled_maps_boxes_and_suppresses_duplicates():
    model = _FakeModel(
        [
            [
                _FakeResult([_FakeBox([170, 20, 260, 120], 0.8)]),
                _FakeResult([_FakeBox([10, 19, 100, 119], 0.9)]),
            ]
        ]
    )

    detections = predict_tiled(
        model,
        Image.new("RGB", (800, 640)),
        tile=640,
        overlap=480,
        batch=2,
        nms_iou=0.4,
    )

    assert detections == [((170, 19, 260, 119), 0.9)]


def test_predict_multiscale_maps_scaled_boxes_back_to_original_coordinates(monkeypatch):
    calls = []

    def fake_predict_tiled(model, image, **kwargs):
        calls.append(image.size)
        return [((20, 30, 60, 90), 0.8)]

    monkeypatch.setattr("src.data.tiling.predict_tiled", fake_predict_tiled)

    detections = predict_multiscale(
        object(),
        Image.new("RGB", (100, 200)),
        scales=(0.5,),
    )

    assert calls == [(50, 100)]
    assert detections == [((40.0, 60.0, 120.0, 180.0), 0.8)]
