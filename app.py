import streamlit as st
from PIL import Image
from utils import detect_hazards, camera_available

st.set_page_config(page_title="HazardVision", layout="wide")
# optional logo display if logo.png is present
try:
    st.image("logo.png", width=140)
except Exception:
    pass

st.title("⚠️ HazardVision — Advanced Spill & Hazard Detector")
st.markdown("""**What this app does (advanced heuristics)**

- Detects liquid spills of many types (milk, water, oil, soda, yogurt, honey, detergent, juices, etc.) using HSV + texture rules.
- Detects hazardous objects in warehouses (boxes, pallets, carts, blocked pathways, blocked panel/extinguisher) using YOLO + proximity heuristics.
- Highlights spills with a semi-transparent overlay and draws color-coded bounding boxes for hazards.
- Placeholder camera support (captures snapshots every 10s) — disabled on Streamlit Cloud; use Docker/local for live camera demos.
""")

st.sidebar.header("Settings")
confidence = st.sidebar.slider("YOLO confidence threshold", 0.05, 0.99, 0.25, 0.01)
min_spill_area = st.sidebar.slider("Min spill area (pixels)", 100, 20000, 400, 100)
st.sidebar.write("Runtime: Streamlit Cloud (Python 3.10) recommended. Docker recommended for camera/live use.")

tabs = st.tabs(["📷 Image Upload", "🎥 Camera (placeholder)", "ℹ️ Notes"])

with tabs[0]:
    uploaded = st.file_uploader("Upload an image (jpg/png)", type=['jpg','jpeg','png'])
    if uploaded is not None:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Uploaded image", use_column_width=True)
        with st.spinner("Running advanced detection (this may take several seconds)..."):
            df, annotated = detect_hazards(image, conf=confidence, min_spill_area=min_spill_area)
        if annotated is not None:
            st.image(annotated, caption="Annotated result", use_column_width=True)
        else:
            st.warning("No annotated image produced (either no detections or rendering not available).")
        st.subheader("Hazard report")
        if df is None or df.empty:
            st.info("No hazards detected at current settings.")
        else:
            st.table(df)

with tabs[1]:
    st.info("Camera placeholder: snapshots every 10s will be supported in the Docker/local version.")
    if camera_available():
        st.success("Camera functionality appears to be available in this environment.")
    else:
        st.warning("Camera functions are not available here (Streamlit Cloud). Use Docker/local to enable camera demo.")

with tabs[2]:
    st.markdown("""**Notes & limitations**

- The spill detector is heuristic-based (color + low-texture + morphology). It covers a wide range of liquids/colors but is not a trained segmentation model — for production-level accuracy you would collect labeled spill images and train a segmentation model.
- The warehouse obstruction heuristics are conservative: they look for objects occupying central walking paths or near edges (possible exit obstruction). Tune `min_spill_area` in the sidebar for sensitivity.
- If you want fully offline startup (no model download), add `yolov8n.pt` to the repo root before deploying.
""")
