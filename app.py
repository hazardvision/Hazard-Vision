import streamlit as st
from PIL import Image
import numpy as np
from utils import detect_hazards, camera_available

st.set_page_config(page_title="HazardVision", layout="wide")

# Optional logo display if file present
try:
    st.image("logo.png", width=140)
except Exception:
    pass

st.title("⚠️ HazardVision — Advanced Spill & Hazard Detector")
st.markdown("""**What this app does (advanced heuristics)**

- Detects a wide variety of spills (milk, water, soda, oil, yogurt, honey, detergent, juices, etc.) using HSV + texture heuristics.
- Detects common warehouse obstructions (boxes, pallets, carts, people blocking central paths) using a pretrained YOLO + proximity heuristics.
- Highlights spills with a semi-transparent overlay and draws color-coded bounding boxes for hazards.
- Camera/live features are placeholders on Streamlit Cloud (use Docker/local for camera demos).
""")

st.sidebar.header("Settings")
confidence = st.sidebar.slider("YOLO confidence threshold", 0.05, 0.99, 0.25, 0.01)
min_spill_area = st.sidebar.slider("Min spill area (pixels)", 100, 20000, 400, 100)
st.sidebar.write("Runtime: Streamlit Cloud (Python 3.10 recommended). Docker recommended for camera.")

tabs = st.tabs(["📷 Image Upload", "🎥 Camera (placeholder)", "ℹ Notes"])

with tabs[0]:
    uploaded = st.file_uploader("Upload an image (jpg/png)", type=['jpg','jpeg','png'])
    if uploaded is not None:
        # Use PIL to open image; pass PIL Image to utils
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Uploaded image (input)", use_column_width=True)
        with st.spinner("Running detection (this may take some seconds)..."):
            try:
                df, annotated = detect_hazards(image, conf=confidence, min_spill_area=min_spill_area)
            except Exception as e:
                # Show a friendly error and log details to Streamlit logs
                st.error("Detection failed. See app logs for details.")
                st.exception(e)
                df, annotated = None, None

        if annotated is not None:
            st.image(annotated, caption="Annotated result", use_column_width=True)
        else:
            st.warning("No annotated image available (no detections or rendering not supported).")

        st.subheader("Hazard report")
        if df is None or df.empty:
            st.info("No hazards detected at current settings.")
        else:
            st.table(df)

with tabs[1]:
    st.info("Camera placeholder: this will capture snapshots every 10s in the Docker/local version.")
    if camera_available():
        st.success("Camera functions appear to be available in this environment.")
    else:
        st.warning("Camera functions are not available on Streamlit Cloud. Use Docker/local for camera demos.")

with tabs[2]:
    st.markdown("""**Notes / Limitations**

- This app uses heuristic spill detection (color + texture + morphology) plus a pretrained YOLOv8n model for object detection. Heuristics greatly improve recall for many liquid types but are not a replacement for a trained segmentation model.
- For production-level claimable accuracy you should collect labeled data and train a segmentation model — the app is structured so you can swap in custom model weights later.
- To avoid cloud binary installation problems the code lazy-loads heavy native packages only when needed; errors (if any) will be visible in the app logs.
""")
