"""
Single entry point for inference: uses real YOLO if configured, else dummy model.
"""
from .config import get_model_path, use_real_yolo

def run_inference(image_path, crop_type="Maize", objective="2"):
    """Run inference; returns dict with detections, metrics, total_count, viable_count, viability_pct, etc."""
    if use_real_yolo():
        try:
            from .yolo_inference import run_inference as real_run
            out = real_run(image_path, crop_type=crop_type, objective=objective, model_path=get_model_path())
            if out is not None:
                return out
        except Exception:
            pass
    from .dummy_yolo import run_inference as dummy_run
    return dummy_run(image_path, crop_type=crop_type, objective=objective)
