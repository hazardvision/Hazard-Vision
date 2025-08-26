import torch
from ultralytics import YOLO
from PIL import Image, ImageDraw

_model = None

def _ensure_model():
    global _model
    if _model is None:
        try:
            _model = YOLO("yolov8n.pt")  # lightweight YOLO
        except Exception:
            import os
            if os.path.exists("yolov8n.pt"):
                os.remove("yolov8n.pt")  # clear corrupted file
            _model = YOLO("yolov8n.pt")
    return _model

def detect_hazards(image, conf=0.5, mode="supermarket"):
    model = _ensure_model()
    results = model.predict(image, conf=conf)

    labels_supermarket = ["water", "milk", "oil", "juice", "detergent", "other spill"]
    labels_warehouse   = ["box", "pallet", "trolley", "clutter", "other obstruction"]

    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0].item())
            conf_score = float(box.conf[0].item())
            xyxy = box.xyxy[0].tolist()
            x1, y1, x2, y2 = map(int, xyxy)

            label = labels_supermarket[cls % len(labels_supermarket)] if mode == "supermarket"                      else labels_warehouse[cls % len(labels_warehouse)]

            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            draw.text((x1, y1 - 10), f"{label} {conf_score:.2f}", fill="red")

    return annotated
