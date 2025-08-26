import streamlit as st
import cv2
import numpy as np
from utils import detect_hazards

st.set_page_config(page_title="HazardVision", layout="wide")

st.title("🛡️ HazardVision – Advanced Spill & Obstruction Detection")

st.sidebar.header("Settings")
mode = st.sidebar.radio("Choose Environment", ["supermarket", "warehouse"])
confidence = st.sidebar.slider("Detection Confidence", 0.2, 1.0, 0.4)

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded:
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    st.subheader("Original Image")
    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    st.subheader("Detected Hazards")
    annotated = detect_hazards(image, conf=confidence, mode=mode)
    st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
else:
    st.info("Upload an image to start hazard detection.")
