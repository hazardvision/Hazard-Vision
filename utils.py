import os
import numpy as np
import pandas as pd
from PIL import Image
from typing import Tuple, Optional, List, Dict

# Lazy cv2 import
def _lazy_cv2():
    try:
        import cv2
        return cv2
    except Exception:
        return None

# Lazy YOLO import
def _lazy_YOLO():
    try:
        from ultralytics import YOLO
        return YOLO
    except Exception as e:
        raise RuntimeError('ultralytics import failed: ' + str(e))

MODEL_PATH = os.getenv('HAZARDVISION_MODEL', 'yolov8n.pt')
_MODEL = None

def _load_model():
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    YOLO = _lazy_YOLO()
    if os.path.exists(MODEL_PATH):
        _MODEL = YOLO(MODEL_PATH)
        return _MODEL
    # Auto-download small model
    tmp = YOLO('yolov8n')
    try:
        tmp.save(MODEL_PATH)
    except Exception:
        pass
    _MODEL = tmp
    return _MODEL

# ---------------- Spill detection heuristics ----------------
def detect_spills_np(img: np.ndarray, min_area:int=400) -> Tuple[np.ndarray, List[List[int]], List[int]]:
    cv2 = _lazy_cv2()
    if cv2 is None:
        return np.zeros(img.shape[:2], dtype='uint8'), [], []

    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    v = hsv[:,:,2].astype(int)
    s = hsv[:,:,1].astype(int)
    hh = hsv[:,:,0].astype(int)

    # heuristics for many liquid types
    spec_mask = (v > 200) & (s < 80)           # specular sheen (clear water)
    milk_mask = (v > 180) & (s < 90)           # milk/yogurt/white
    dark_mask = (v < 120) & (s < 120)          # dark liquids like cola
    oil_mask = ((hh >= 5) & (hh <= 45)) & (s > 50) & (v > 100)  # oil-ish
    color_mask = (s > 60) & (v > 90)           # colored detergents/juices

    raw = (spec_mask | milk_mask | dark_mask | oil_mask | color_mask).astype('uint8') * 255

    # low texture filter (spills are smoother)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap_var = cv2.convertScaleAbs(cv2.normalize(np.abs(lap), None, 0, 255, cv2.NORM_MINMAX))
    low_texture = lap_var < 20
    raw = cv2.bitwise_and(raw, (low_texture.astype('uint8')*255))

    # morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    clean = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, kernel, iterations=2)
    clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_final = np.zeros((h,w), dtype='uint8')
    bboxes = []
    areas = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        x,y,ww,hh = cv2.boundingRect(c)
        if ww < 10 or hh < 10:
            continue
        cv2.drawContours(mask_final, [c], -1, 255, -1)
        bboxes.append([int(x),int(y),int(x+ww),int(y+hh)])
        areas.append(int(area))
    return mask_final, bboxes, areas

# ---------------- Pathway / obstruction heuristics ----------------
def analyze_obstruction(objects: List[Dict], img_shape:Tuple[int,int], center_strip_ratio:float=0.3) -> List[Dict]:
    h, w = img_shape
    csw = int(w * center_strip_ratio)
    cx1 = (w - csw)//2
    cx2 = cx1 + csw
    issues = []
    for obj in objects:
        x1,y1,x2,y2 = obj['bbox']
        inter_x1 = max(x1, cx1); inter_x2 = min(x2, cx2)
        inter_w = max(0, inter_x2 - inter_x1)
        obj_w = x2 - x1
        if obj_w > 0 and (inter_w / obj_w) > 0.5:
            issues.append({'type':'pathway_blockage', 'name': obj['name'], 'bbox':obj['bbox'], 'conf':obj['conf']})
        if x1 < int(0.05*w) and (x2-x1) > 0.15*w:
            issues.append({'type':'edge_obstruction', 'name': obj['name'], 'bbox':obj['bbox'], 'conf':obj['conf']})
        if x2 > int(0.95*w) and (x2-x1) > 0.15*w:
            issues.append({'type':'edge_obstruction', 'name': obj['name'], 'bbox':obj['bbox'], 'conf':obj['conf']})
    return issues

# ---------------- Main combined detection ----------------
def detect_hazards(pil_image: Image.Image, conf:float=0.25, min_spill_area:int=400):
    cv2 = _lazy_cv2()
    model = _load_model()
    img = np.array(pil_image)  # RGB
    h, w = img.shape[:2]

    # YOLO detections
    results = model.predict(img, conf=conf, verbose=False)
    objects = []
    annotated = img.copy()
    if results and len(results)>0:
        r = results[0]
        names = getattr(r, 'names', {}) or {}
        boxes = getattr(r, 'boxes', None)
        if boxes is not None:
            # try multiple access patterns for compatibility
            try:
                xyxy = boxes.xyxy.cpu().numpy()
                cls_ids = boxes.cls.cpu().numpy().astype(int)
                confs = boxes.conf.cpu().numpy()
            except Exception:
                # fallback if boxes data structure differs
                xyxy = getattr(boxes, 'xyxy', None)
                cls_ids = getattr(boxes, 'cls', None)
                confs = getattr(boxes, 'conf', None)
                if xyxy is None:
                    xyxy = []
                    cls_ids = []
                    confs = []
            for b, cid, sc in zip(xyxy, cls_ids, confs):
                x1,y1,x2,y2 = map(int, b)
                name = names.get(int(cid), str(cid))
                objects.append({'name':name, 'bbox':[x1,y1,x2,y2], 'conf': float(sc)})
                if cv2 is not None:
                    cv2.rectangle(annotated, (x1,y1), (x2,y2), (0,255,0), 2)
                    cv2.putText(annotated, f"{name}:{sc:.2f}", (x1, max(15,y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    # analyze obstructions
    issues = analyze_obstruction(objects, (h,w), center_strip_ratio=0.3)
    for it in issues:
        bx = it['bbox']; x1,y1,x2,y2 = bx
        if cv2 is not None:
            cv2.rectangle(annotated, (x1,y1), (x2,y2), (0,0,255), 3)
            cv2.putText(annotated, it['type'], (x1, max(15,y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

    # detect spills
    spill_mask, spill_boxes, spill_areas = detect_spills_np(img, min_area=min_spill_area)
    detections = []
    if spill_mask.sum() > 0 and cv2 is not None:
        overlay = annotated.copy()
        overlay[spill_mask>0] = (255,0,0)
        alpha = 0.45
        annotated = cv2.addWeighted(overlay.astype('uint8'), alpha, annotated.astype('uint8'), 1-alpha, 0)
        for i, box in enumerate(spill_boxes):
            x1,y1,x2,y2 = box
            area = spill_areas[i] if i < len(spill_areas) else 0
            if cv2 is not None:
                cv2.rectangle(annotated, (x1,y1), (x2,y2), (0,0,255), 2)
                cv2.putText(annotated, f"Spill area={area}", (x1, max(15,y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
            detections.append({'Hazard':'spill', 'Confidence':0.9, 'Type':'spill', 'bbox':[x1,y1,x2,y2], 'area':area})

    # include YOLO objects and issues
    for obj in objects:
        detections.append({'Hazard': obj['name'], 'Confidence': obj['conf'], 'Type':'object', 'bbox': obj['bbox']})
    for it in issues:
        detections.append({'Hazard': it['type'], 'Confidence': it.get('conf', 0.9), 'Type':'issue', 'bbox': it['bbox']})

    try:
        annotated_pil = Image.fromarray(annotated.astype('uint8'))
    except Exception:
        annotated_pil = None

    df = pd.DataFrame(detections)
    return df, annotated_pil
