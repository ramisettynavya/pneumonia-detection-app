import streamlit as st
from PIL import Image
import inference
import numpy as np

def looks_like_xray(img):
    # 1. X-rays are grayscale
    a = np.asarray(img.resize((128, 128))).astype(float)
    colour_diff = (np.abs(a[..., 0] - a[..., 1]) + np.abs(a[..., 1] - a[..., 2])).mean()
    if colour_diff >= 12:
        return False
    # 2. X-rays are roughly square-ish, not panoramic or tall strips
    w, h = img.size
    return 0.7 <= w / h <= 1.8

@st.cache_resource
def get_model():
    return inference.load_model("vit_b16_best.pth")

model = get_model()

st.title("Explainable Pneumonia Detection from Chest X-rays")
st.caption("Vision Transformer (ViT-B/16) with attention-based explanation")
st.warning("Research prototype for decision support only. Not a medical diagnosis. Please consult a qualified doctor.")

file = st.file_uploader("Upload a chest X-ray image", type=["jpg", "jpeg", "png"])
if file is not None:
    try:
        img = Image.open(file).convert("RGB")
    except Exception:
        st.error("This file could not be read as an image. Please upload a JPG or PNG.")
        st.stop()

    if not looks_like_xray(img):
        st.error("This does not look like a chest X-ray, because it contains colour. "
                 "Please upload a grayscale chest X-ray image.")
        st.stop()

    with st.spinner("Analysing the X-ray..."):
        result = inference.analyze(model, img)

    col1, col2 = st.columns(2)
    col1.subheader("Uploaded X-ray")
    col1.image(img, use_container_width=True)
    col2.subheader("Where the model focused")
    col2.image(result["overlay"], use_container_width=True)

    st.subheader("Result")
    conf = result["confidence"]
    if conf < 0.90:
        color, bg, title = "#B45309", "#FEF3C7", "UNCERTAIN RESULT"
    elif result["label"] == "PNEUMONIA":
        color, bg, title = "#B91C1C", "#FEE2E2", "PNEUMONIA DETECTED"
    else:
        color, bg, title = "#047857", "#D1FAE5", "NO PNEUMONIA DETECTED"

    st.markdown(
        f"""<div style="background:{bg};border-left:8px solid {color};
        padding:16px 22px;border-radius:10px;margin-bottom:8px;">
        <div style="color:{color};font-size:28px;font-weight:700;">{title}</div>
        <div style="color:#374151;font-size:16px;">Model confidence: {conf:.2%}</div>
        </div>""",
        unsafe_allow_html=True,
    )
    st.progress(float(conf))
    with st.container(border=True):
        st.markdown(result["report"])
else:
    st.info("Upload an X-ray to see the prediction, heatmap and explanation.")