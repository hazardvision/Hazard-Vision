import streamlit as st
from utils import detect_hazards

st.set_page_config(page_title="HazardVision", layout="wide")

st.title("🛡️ HazardVision - Spill & Obstruction Detector")

st.markdown("**Modes:** Supermarkets (liquid spills) | Warehouses (obstructions).")

mode = st.selectbox("Choose mode:", ["supermarket", "warehouse"])
confidence = st.slider("Confidence threshold", 0.1, 1.0, 0.5, 0.05)

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded:
    import PIL.Image as Image
    image = Image.open(uploaded).convert("RGB")

    st.image(image, caption="Uploaded Image", use_column_width=True)

    annotated = detect_hazards(image, conf=confidence, mode=mode)
    st.image(annotated, caption="Detection Results", use_column_width=True)
