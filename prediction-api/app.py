import io
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from flask import Flask, request, jsonify

CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR))

MODELS_DIR = CURRENT_DIR / "models"

from src.data.patch_generation import PATCH_SIZE
from src.eval.box_utils import non_max_suppression
from src.models.patch_classifier import PatchClassifier
from src.models.sliding_window import DEFAULT_WINDOW_SIZES, windows_for_scale
from src.data.patch_dataset import to_tensor


app = Flask(__name__)


# Model Loading
def load_sliding_window_model() -> PatchClassifier | None:
    path = MODELS_DIR / "sliding_window_classifier.pt"
    if not path.exists():
        return None
    model = PatchClassifier()
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model

sw_model = load_sliding_window_model()

# Inference
def run_sliding_window(model: PatchClassifier, image: Image.Image, score_threshold: float):
    W, H = image.width, image.height
    all_boxes, all_scores = [], []

    for size in DEFAULT_WINDOW_SIZES:
        stride = max(1, int(size * 0.5))
        boxes = windows_for_scale(W, H, size, stride)
        batch_size = 128

        for start in range(0, len(boxes), batch_size):
            batch = boxes[start: start + batch_size]
            crops = [image.crop(b).resize((PATCH_SIZE, PATCH_SIZE), Image.BILINEAR) for b in batch]
            tensors = torch.stack([to_tensor(np.array(c.convert("RGB"))) for c in crops])

            with torch.no_grad():
                scores = torch.sigmoid(model(tensors)).numpy()

            for box, score in zip(batch, scores):
                if score >= score_threshold:
                    all_boxes.append(box)
                    all_scores.append(float(score))

    if not all_boxes:
        return []
    keep_idx = non_max_suppression(np.array(all_boxes), np.array(all_scores), iou_threshold=0.2)
    return [(tuple(all_boxes[i]), all_scores[i]) for i in keep_idx]


# API Endpoints
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["file"]
    image = Image.open(io.BytesIO(file.read())).convert("RGB")

    model_choice = request.form.get("model_choice")
    conf_threshold = float(request.form.get("conf_threshold", 0.05))
    top_n = int(request.form.get("top_n", 10))

    if model_choice == "sliding_window" and sw_model:
        raw_detections = run_sliding_window(sw_model, image, conf_threshold)
        detections = sorted(raw_detections, key=lambda d: -d[1])[:top_n]
    else:
        return jsonify({"error": "Invalid model choice or model not loaded. Model loaded: sliding_window."}), 400

    # Format output for JSON serialization
    results = [{"box": box, "score": score} for box, score in detections]
    return jsonify({"detections": results})


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)