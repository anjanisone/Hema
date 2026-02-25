"""
Real YOLO inference using Ultralytics (YOLOv8 detection or segmentation).
Use when DEEPSPOT_MODEL_PATH points to a .pt weights file; otherwise the app uses the dummy model.
Expects model classes: 0 = viable, 1 = unviable (or compatible mapping).
"""
import time
from pathlib import Path

try:
    from ultralytics import YOLO
    HAS_ULTRALYTICS = True
except ImportError:
    HAS_ULTRALYTICS = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


def _metrics_from_detections(detections):
    """Compute morphology-style metrics from list of {box, class} (same shape as dummy)."""
    if not detections:
        return {"avg_diameter_um": 0, "avg_area_um2": 0, "avg_perimeter_um": 0,
                "avg_perimeter_sqrt_area": 0, "avg_circularity": 0, "avg_aspect_ratio": 0}
    viable = [d for d in detections if d["class"] == 0]
    all_d = detections

    def stats(dets):
        if not detets:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        areas, perims, aspect_ratios = [], [], []
        for d in detets:
            x1, y1, x2, y2 = d["box"]
            w, h = x2 - x1, y2 - y1
            area = w * h
            perim = 2 * (w + h)
            areas.append(area)
            perims.append(perim)
            aspect_ratios.append(max(w, h) / max(min(w, h), 1))
        n = len(dets)
        avg_area = sum(areas) / n
        avg_perim = sum(perims) / n
        scale = 0.5
        avg_diam = (avg_perim / 3.14159) * scale
        avg_area_um = avg_area * scale * scale
        avg_perim_um = avg_perim * scale
        sqrt_a = avg_area_um ** 0.5
        circ = (4 * 3.14159 * avg_area_um / (avg_perim_um ** 2)) if avg_perim_um else 0
        circ = min(1.0, max(0, circ))
        return avg_diam, avg_area_um, avg_perim_um, avg_perim_um / sqrt_a if sqrt_a else 0, circ, sum(aspect_ratios) / n

    v_diam, v_area, v_perim, v_psa, v_circ, v_ar = stats(viable)
    a_diam, a_area, a_perim, a_psa, a_circ, a_ar = stats(all_d)
    return {
        "viable": {"avg_diameter_um": v_diam, "avg_area_um2": v_area, "avg_perimeter_um": v_perim,
                   "avg_perimeter_sqrt_area": v_psa, "avg_circularity": v_circ, "avg_aspect_ratio": v_ar},
        "all": {"avg_diameter_um": a_diam, "avg_area_um2": a_area, "avg_perimeter_um": a_perim,
                "avg_perimeter_sqrt_area": a_psa, "avg_circularity": a_circ, "avg_aspect_ratio": a_ar},
    }


def run_inference(image_path, crop_type="Maize", objective="2", model_path=None):
    """
    Run real YOLO inference. Returns same dict as dummy_yolo.run_inference.
    model_path: path to .pt file (if None, caller must ensure it's set via config).
    """
    if not HAS_ULTRALYTICS or not model_path:
        return None
    start = time.perf_counter()
    img_w, img_h = 640, 480
    if HAS_PIL:
        try:
            with Image.open(image_path) as im:
                img_w, img_h = im.size
        except Exception:
            pass

    model = YOLO(model_path)
    results = model.predict(source=image_path, verbose=False)
    detections = []
    for result in results:
        if result.boxes is None:
            continue
        boxes = result.boxes
        xyxy = boxes.xyxy.cpu().numpy()
        cls = boxes.cls.cpu().numpy()
        conf = boxes.conf.cpu().numpy() if boxes.conf is not None else [0.9] * len(cls)
        for i in range(len(xyxy)):
            x1, y1, x2, y2 = map(int, xyxy[i])
            c = int(cls[i])
            if c > 1:
                c = 1
            detections.append({"box": (x1, y1, x2, y2), "class": c, "confidence": float(conf[i]) if i < len(conf) else 0.9})

    metrics = _metrics_from_detections(detections)
    if not detections:
        metrics = {
            "viable": {"avg_diameter_um": 0, "avg_area_um2": 0, "avg_perimeter_um": 0, "avg_perimeter_sqrt_area": 0, "avg_circularity": 0, "avg_aspect_ratio": 0},
            "all": {"avg_diameter_um": 0, "avg_area_um2": 0, "avg_perimeter_um": 0, "avg_perimeter_sqrt_area": 0, "avg_circularity": 0, "avg_aspect_ratio": 0},
        }
    elapsed_ms = (time.perf_counter() - start) * 1000
    viable_count = sum(1 for d in detections if d["class"] == 0)
    total = len(detections)
    viability_pct = (100.0 * viable_count / total) if total else 0.0
    return {
        "detections": detections,
        "metrics": metrics,
        "inference_time_ms": round(elapsed_ms, 2),
        "total_count": total,
        "viable_count": viable_count,
        "viability_pct": round(viability_pct, 1),
        "image_size": (img_w, img_h),
    }
