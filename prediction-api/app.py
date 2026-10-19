import io
import json
import sys
from pathlib import Path
import base64

import numpy as np
import torch
from PIL import Image
from flask import Flask, request, jsonify

CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR))

MODELS_DIR = CURRENT_DIR / "models"

from src.data.patch_generation import PATCH_SIZE
from src.data.tiling import DEFAULT_SCALES, predict_multiscale
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


def load_yolo_model():
    path = MODELS_DIR / "yolo_detector.pt"
    if not path.exists():
        return None
    from ultralytics import YOLO
    return YOLO(str(path))


def load_tiled_yolo():
    path = MODELS_DIR / "yolo_tiled_detector.pt"
    if not path.exists():
        return None, {}
    from ultralytics import YOLO
    config_path = MODELS_DIR / "yolo_tiled_config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    return YOLO(str(path)), config


sw_model = load_sliding_window_model()
yolo_model = load_yolo_model()
tiled_model, tiled_config = load_tiled_yolo()


# Inference

def run_sliding_window(model: PatchClassifier, image: Image.Image, score_threshold: float):
    W, H = image.width, image.height
    heat = np.zeros((H, W), dtype=np.float32)
    heat_count = np.zeros((H, W), dtype=np.float32)

    all_boxes, all_scores = [], []
    frames = []

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
                x1, y1, x2, y2 = box
                heat[y1:y2, x1:x2] += score
                heat_count[y1:y2, x1:x2] += 1
                if score >= score_threshold:
                    all_boxes.append(box)
                    all_scores.append(float(score))

            # Generate the heatmap frame
            avg_heat = np.divide(heat, heat_count, out=np.zeros_like(heat), where=heat_count > 0)
            overlay = Image.fromarray((avg_heat * 255).astype(np.uint8)).convert("L")
            heat_rgb = Image.merge("RGB", (overlay, Image.new("L", overlay.size, 0), Image.new("L", overlay.size, 0)))
            blended = Image.blend(image.convert("RGB"), heat_rgb, alpha=0.5)
            frames.append(blended)

    # Perform Non-Max Suppression
    if not all_boxes:
        keep_detections = []
    else:
        keep_idx = non_max_suppression(np.array(all_boxes), np.array(all_scores), iou_threshold=0.2)
        keep_detections = [(tuple(all_boxes[i]), all_scores[i]) for i in keep_idx]

    # Compile frames into a base64 GIF
    gif_b64 = None
    if frames:
        gif_io = io.BytesIO()
        frames[0].save(
            gif_io,
            format="GIF",
            save_all=True,
            append_images=frames[1:],
            duration=100,  # 100ms per frame
            loop=0
        )
        gif_b64 = base64.b64encode(gif_io.getvalue()).decode("utf-8")

    return keep_detections, gif_b64


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
    gif_b64 = None

    if model_choice == "tiled" and tiled_model:
        scales = tuple(tiled_config.get("scales", DEFAULT_SCALES))
        raw_detections = predict_multiscale(tiled_model, image, scales=scales, conf=0.01)
        detections = [d for d in raw_detections if d[1] >= conf_threshold][:top_n]

    elif model_choice == "sliding_window" and sw_model:
        raw_detections, gif_b64 = run_sliding_window(sw_model, image, conf_threshold)
        detections = sorted(raw_detections, key=lambda d: -d[1])[:top_n]

    elif model_choice == "yolo" and yolo_model:
        yolo_result = yolo_model.predict(image, conf=conf_threshold, verbose=False)[0]
        raw_detections = ((tuple(b.xyxy[0].tolist()), float(b.conf.item())) for b in yolo_result.boxes)
        detections = sorted(raw_detections, key=lambda d: -d[1])[:top_n]

    else:
        return jsonify({"error": "Invalid model choice or model not loaded. Models loaded: tiled, sliding_window, yolo"}), 400

    # Format output for JSON serialization
    results = [{"box": box, "score": score} for box, score in detections]
    return jsonify({"detections": results, "animation_b64": gif_b64})


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)