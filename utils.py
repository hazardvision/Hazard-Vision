import cv2
import numpy as np
from ultralytics import YOLO

# Lazy load YOLO model
_model = None
def _ensure_model():
    global _model
    if _model is None:
        _model = YOLO("yolov8n.pt")  # lightweight model
    return _model

# Hazard categories
SUPERMARKET_SPILLS = ["water", "milk", "soda", "oil", "detergent", "yogurt", "honey"]
WAREHOUSE_OBSTRUCTIONS = ["box", "pallet", "ladder", "cart", "spill", "debris"]

def detect_hazards(image, conf=0.4, mode="supermarket"):
    model = _ensure_model()
    results = model.predict(image, conf=conf)

    annotated = image.copy()

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf_score = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = model.names[cls_id]

            if mode == "supermarket":
                hazard_type = label if label in SUPERMARKET_SPILLS else "spill"
            else:
                hazard_type = label if label in WAREHOUSE_OBSTRUCTIONS else "obstruction"

            # Draw bounding rectangle & label
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(
                annotated,
                f"{hazard_type} ({conf_score:.2f})",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
    return annotated
