"""
Dummy YOLO model for internal testing. No torch/ultralytics required.
Returns deterministic fake detections and metrics so the UI can be exercised.
"""
import time
import hashlib
from pathlib import Path

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

# Seed detections from image path hash so same image gives same "results"
def _seed_from_path(path):
    s = str(Path(path).resolve())
    return int(hashlib.md5(s.encode()).hexdigest()[:8], 16)

def _fake_detections(n, seed, img_w, img_h):
    """Generate n fake boxes and labels (0=viable, 1=unviable)."""
    # Deterministic "random" from seed
    state = seed
    def next_():
        nonlocal state
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF
        return state / 0x7FFFFFFF
    dets = []
    margin = min(80, img_w // 8, img_h // 8)
    for i in range(n):
        w = 30 + int(next_() * 40)
        h = 28 + int(next_() * 44)
        x1 = margin + int(next_() * (img_w - 2 * margin - w))
        y1 = margin + int(next_() * (img_h - 2 * margin - h))
        x2, y2 = x1 + w, y1 + h
        # ~60% viable for demo
        cls = 0 if next_() < 0.6 else 1
        conf = 0.7 + next_() * 0.25
        dets.append({"box": (x1, y1, x2, y2), "class": cls, "confidence": conf})
    return dets

def _metrics(detections):
    """Compute fake morphology metrics from detections."""
    if not detections:
        return {"avg_diameter_um": 0, "avg_area_um2": 0, "avg_perimeter_um": 0,
                "avg_perimeter_sqrt_area": 0, "avg_circularity": 0, "avg_aspect_ratio": 0}
    viable = [d for d in detections if d["class"] == 0]
    all_d = detections
    def stats(dets):
        if not dets:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        areas = []
        perims = []
        aspect_ratios = []
        for d in dets:
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
        # Scale to "µm" for display (arbitrary scale)
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

def run_inference(image_path, crop_type="Maize", objective="2"):
    """
    Dummy inference: load image size, return fake detections and metrics.
    Returns dict: detections (list of {box, class, confidence}), metrics, inference_time_ms, count, viable_count, viability_pct.
    """
    start = time.perf_counter()
    img_w, img_h = 640, 480
    if HAS_PIL:
        try:
            with Image.open(image_path) as im:
                img_w, img_h = im.size
        except Exception:
            pass
    seed = _seed_from_path(image_path)
    # Number of "pollen" between 5 and 20
    n = 5 + (seed % 16)
    detections = _fake_detections(n, seed, img_w, img_h)
    metrics = _metrics(detections)
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
