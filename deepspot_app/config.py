"""
DeepSpot config. SQLite on network drive; optional real YOLO model path.
"""
import os

# --- Database (network drive for shared history) ---
DEFAULT_DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DEFAULT_DB_DIR, exist_ok=True)
DEFAULT_DB_PATH = os.path.join(DEFAULT_DB_DIR, "deepspot.db")

def get_db_path():
    return os.environ.get("DEEPSPOT_DB_PATH", DEFAULT_DB_PATH)

def get_results_dir():
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(d, exist_ok=True)
    return d

# --- YOLO model (real model weights; omit to use dummy) ---
# Path to .pt weights, e.g. DeepSpot.pt or yolov8n-seg.pt
#   Windows: set DEEPSPOT_MODEL_PATH=C:\Models\DeepSpot.pt
#   macOS:   export DEEPSPOT_MODEL_PATH=/path/to/DeepSpot.pt
DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
DEFAULT_MODEL_PATH = os.path.join(DEFAULT_MODEL_DIR, "DeepSpot.pt")

def get_model_path():
    """Path to .pt weights, or None to use dummy model."""
    p = os.environ.get("DEEPSPOT_MODEL_PATH", "").strip()
    if p and os.path.isfile(p):
        return p
    if os.path.isfile(DEFAULT_MODEL_PATH):
        return DEFAULT_MODEL_PATH
    return None

def use_real_yolo():
    """True if a real YOLO model path is set and the file exists."""
    return get_model_path() is not None
