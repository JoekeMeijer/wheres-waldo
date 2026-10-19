import base64
import os
import io
import requests
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

from waldo_theme import (
    apply_theme_html,
    hero_html,
    hidden_waldo_html,
    stripes_divider_html,
    verdict_html,
)

# Pull the API URL from environment variables, matching the lab setup
PREDICTOR_API = os.environ.get("PREDICTOR_API", "http://localhost:5000/predict")

st.set_page_config(page_title="Where's Waldo?", page_icon="🔍", layout="wide")
st.markdown(apply_theme_html(), unsafe_allow_html=True)
st.markdown(hero_html(), unsafe_allow_html=True)
st.caption(
    "Three ways to find Waldo. The inference logic is now running securely on a backend API."
)


# Helpers

def _font(size: int) -> ImageFont.ImageFont:
    for name in ("arialbd.ttf", "arial.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size)


def draw_boxes(image: Image.Image, boxes_with_scores: list, color: str, label_top: int | None = None) -> Image.Image:
    out = image.copy()
    draw = ImageDraw.Draw(out)
    line = max(4, image.width // 250)
    font = _font(max(18, image.width // 55))

    for rank, (box, score) in enumerate(boxes_with_scores, start=1):
        if label_top is not None and rank > label_top:
            draw.rectangle(box, outline=color, width=max(2, line // 2))
            continue
        draw.rectangle(box, outline=color, width=line)
        tag = f"#{rank}  {score * 100:.0f}%" if score is not None else f"#{rank}"
        left, top, right, bottom = draw.textbbox((0, 0), tag, font=font)
        pad = 4
        tag_x = min(max(0, box[0]), max(0, image.width - (right - left) - 2 * pad))
        tag_y = box[1] - (bottom - top) - 2 * pad - line
        if tag_y < 0:
            tag_y = box[3] + line
        draw.rectangle((tag_x, tag_y, tag_x + right - left + 2 * pad, tag_y + bottom - top + 2 * pad), fill=color)
        draw.text((tag_x + pad - left, tag_y + pad - top), tag, fill="white", font=font)
    return out


def zoomed_crop(image: Image.Image, box: tuple, out_px: int = 360) -> Image.Image:
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = min(max(x2 - x1, y2 - y1, 40) * 3, image.width, image.height)
    left = min(max(0, cx - side / 2), image.width - side)
    top = min(max(0, cy - side / 2), image.height - side)
    crop = image.crop((round(left), round(top), round(left + side), round(top + side))).resize(
        (out_px, out_px), Image.LANCZOS
    )
    scale = out_px / side
    ImageDraw.Draw(crop).rectangle(
        ((x1 - left) * scale, (y1 - top) * scale, (x2 - left) * scale, (y2 - top) * scale), outline="red", width=2
    )
    return crop


def show_candidates(image: Image.Image, detections: list, max_show: int = 10) -> None:
    if not detections:
        return
    shown = detections[:max_show]
    st.markdown(stripes_divider_html(), unsafe_allow_html=True)
    st.subheader("The usual suspects, up close")
    st.caption("Look for the red-and-white striped shirt and bobble hat, round glasses and a cane.")

    layouts = [[2, 1, 1]] + [[1, 1, 1, 1]] * ((len(shown) - 3 + 3) // 4)
    rank = 0
    for spec in layouts:
        for col in st.columns(spec):
            if rank >= len(shown):
                break
            box, score = shown[rank]
            with col:
                st.image(
                    zoomed_crop(image, box, out_px=640 if rank == 0 else 420),
                    caption=f"#{rank + 1} — {score * 100:.0f}% confident" + (" (best guess)" if rank == 0 else ""),
                    use_container_width=True,
                )
            rank += 1


# UI

uploaded = st.file_uploader("Drop in a page from a Where's Waldo book", type=["jpg", "jpeg", "png"])

model_map = {
    "YOLO on native-resolution tiles": "tiled",
    "Sliding-window CNN (from scratch)": "sliding_window",
    "YOLO on the whole shrunken scene": "yolo"
}

model_display_choice = st.radio("Who should do the looking?", list(model_map.keys()), horizontal=True)

if uploaded is not None and model_display_choice is not None:
    image = Image.open(uploaded).convert("RGB")
    shown_detections = []

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("The page")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("The search")
        placeholder = st.empty()
        model_choice = model_map[model_display_choice]

        # Configure UI parameters based on selected model
        if model_choice == "tiled":
            conf = st.slider("Minimum confidence", 0.01, 0.95, 0.05, 0.01)
            top_n = st.slider("Show at most this many candidates", 1, 10, 3)
        elif model_choice == "sliding_window":
            conf = st.slider("Detection confidence threshold", 0.1, 0.95, 0.6, 0.05)
            top_n = 10
        else:
            conf = st.slider("Detection confidence threshold", 0.05, 0.95, 0.1, 0.05)
            top_n = 10

        if st.button("Find Waldo!", type="primary"):
            with st.spinner("Peering into the crowd..."):
                # Send the image and parameters to the API
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format='JPEG')
                img_byte_arr.seek(0)

                payload = {
                    "model_choice": model_choice,
                    "conf_threshold": conf,
                    "top_n": top_n
                }

                try:
                    response = requests.post(PREDICTOR_API, files={"file": ("image.jpg", img_byte_arr, "image/jpeg")}, data=payload)

                    if response.status_code != 200:
                        st.error(f"API Error: {response.text}")

                    response.raise_for_status()

                    # Parse the results from the API
                    results = response.json().get("detections", [])
                    detections = [(tuple(d["box"]), d["score"]) for d in results]
                    animation_b64 = response.json().get("animation_b64")

                    if detections:
                        st.markdown(verdict_html(detections[0][1]), unsafe_allow_html=True)

                        if animation_b64:
                            gif_bytes = base64.b64decode(animation_b64)
                            st.image(gif_bytes, caption="CNN Heatmap Progression", use_container_width=True)

                        placeholder.image(
                            draw_boxes(image, detections, color="red",
                                       label_top=5 if model_choice == "sliding_window" else None),
                            caption=f"{len(detections)} candidate(s) returned",
                            use_container_width=True,
                        )
                        shown_detections = detections
                    else:
                        st.markdown(verdict_html(None), unsafe_allow_html=True)

                except requests.exceptions.RequestException as e:
                    st.error(f"Error connecting to Prediction API: {e}")

    show_candidates(image, shown_detections)

st.markdown(hidden_waldo_html(), unsafe_allow_html=True)