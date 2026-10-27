import importlib.util
import io
from pathlib import Path

import pytest
from PIL import Image


@pytest.fixture
def api_module():
    path = Path(__file__).parents[1] / "prediction-api" / "app.py"
    spec = importlib.util.spec_from_file_location("prediction_api_app", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _image_upload():
    image = io.BytesIO()
    Image.new("RGB", (32, 24), "white").save(image, format="PNG")
    image.seek(0)
    return {"file": (image, "page.png")}


def test_predict_requires_an_image(api_module):
    api_module.app.config["TESTING"] = True
    response = api_module.app.test_client().post("/predict", data={"model_choice": "tiled"})

    assert response.status_code == 400
    assert response.get_json() == {"error": "No image file provided"}


def test_predict_rejects_unavailable_model(api_module):
    api_module.app.config["TESTING"] = True
    api_module.tiled_model = None
    response = api_module.app.test_client().post("/predict", data=_image_upload(), content_type="multipart/form-data")

    assert response.status_code == 400
    assert "Invalid model choice" in response.get_json()["error"]


def test_predict_returns_filtered_top_n_tiled_detections(api_module, monkeypatch):
    api_module.app.config["TESTING"] = True
    api_module.tiled_model = object()
    api_module.tiled_config = {"scales": [1.0]}
    monkeypatch.setattr(
        api_module,
        "predict_multiscale",
        lambda *args, **kwargs: [
            ((1, 2, 10, 12), 0.4),
            ((3, 4, 11, 14), 0.9),
            ((5, 6, 12, 15), 0.7),
        ],
    )

    response = api_module.app.test_client().post(
        "/predict",
        data={**_image_upload(), "model_choice": "tiled", "conf_threshold": "0.5", "top_n": "1"},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    assert response.get_json() == {
        "detections": [{"box": [3, 4, 11, 14], "score": 0.9}]
    }


def test_predict_returns_sliding_window_detections_in_score_order(api_module, monkeypatch):
    api_module.app.config["TESTING"] = True
    api_module.sw_model = object()
    monkeypatch.setattr(
        api_module,
        "run_sliding_window",
        lambda model, image, threshold: [((0, 0, 4, 4), 0.3), ((5, 5, 9, 9), 0.8)],
    )

    response = api_module.app.test_client().post(
        "/predict",
        data={**_image_upload(), "model_choice": "sliding_window", "conf_threshold": "0.2", "top_n": "2"},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    assert response.get_json()["detections"] == [
        {"box": [5, 5, 9, 9], "score": 0.8},
        {"box": [0, 0, 4, 4], "score": 0.3},
    ]
