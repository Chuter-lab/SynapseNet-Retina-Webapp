"""Configuration for SynapseNet-Retina webapp."""
import os
from pathlib import Path

# Paths
DATA_ROOT = Path("/data/thiru")
MODELS_DIR = DATA_ROOT / "models"
UPLOAD_DIR = DATA_ROOT / "temp" / "uploads"
TEMP_DIR = DATA_ROOT / "temp"
LOG_DIR = DATA_ROOT / "logs"

# Model checkpoints — per-structure list (ensemble uses multiple)
CHECKPOINTS = {
    "mitochondria": [
        MODELS_DIR / "retina_mitochondria_best.pt",       # run008 focal fine-tuned
        MODELS_DIR / "retina_mitochondria_da.pt",          # domain adapted
        MODELS_DIR / "retina_mitochondria_da_ft.pt",       # DA then fine-tuned
    ],
    "membrane": [
        MODELS_DIR / "retina_membrane_best.pt",            # run008 focal fine-tuned
    ],
    "ribbon": [
        MODELS_DIR / "retina_ribbon_best.pt",              # focal fine-tuned
    ],
}

# Model architecture
MODEL_PARAMS = {
    "in_channels": 1,
    "out_channels": 1,
    "depth": 4,
    "initial_features": 32,
    "final_activation": None,  # focal loss models have no built-in activation
}

# Inference — matches evaluation pipeline on node01
DEFAULT_THRESHOLD = 0.5
TILE_SIZE = 256       # patch size used during training and evaluation
TILE_STRIDE = 192     # 75% of patch size (same as eval)
GPU_ID = 0

# TTA and ensemble settings (matching best combination optimization results)
TTA_AUGMENTATIONS = {
    "mitochondria": True,   # 7 geometric augmentations
    "membrane": False,       # no TTA for best membrane result
    "ribbon": False,         # TBD — updated after training evaluation
}

ENSEMBLE_METHOD = {
    "mitochondria": "max",  # element-wise max of prob maps from 3 models
    "membrane": None,        # single model, no ensemble
    "ribbon": None,          # single model, no ensemble
}

# Structures and colors (BGR for OpenCV)
# Per-structure thresholds from combination optimization on retina validation data
STRUCTURES = {
    "mitochondria": {"color": (0, 255, 0), "label": "Mitochondria", "min_area": 5000, "threshold": 0.35},
    "membrane": {"color": (255, 0, 0), "label": "Presynaptic Membrane", "min_area": 50000, "threshold": 0.4},
    "ribbon": {"color": (255, 0, 255), "label": "Synaptic Ribbon", "min_area": 100, "threshold": 0.8},
}

# Overlay
OVERLAY_ALPHA = 0.35

# Web
APP_HOST = "0.0.0.0"
APP_PORT = 7861
MAX_DISPLAY_SIZE = 2048  # Resize for Gradio display

# Auth — loaded from THIRU_AUTH env var: "user1:pass1,user2:pass2"
def _parse_auth():
    auth_str = os.environ.get("THIRU_AUTH", "")
    if not auth_str:
        return None
    pairs = []
    for entry in auth_str.split(","):
        entry = entry.strip()
        if ":" in entry:
            user, passwd = entry.split(":", 1)
            pairs.append((user.strip(), passwd.strip()))
    return pairs or None

AUTH_CREDENTIALS = _parse_auth()
