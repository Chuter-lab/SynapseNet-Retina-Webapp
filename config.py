"""Configuration for SynapseNet-Retina webapp."""
from pathlib import Path

# Paths
DATA_ROOT = Path("/data/thiru")
MODELS_DIR = DATA_ROOT / "models"
UPLOAD_DIR = DATA_ROOT / "temp" / "uploads"
TEMP_DIR = DATA_ROOT / "temp"
LOG_DIR = DATA_ROOT / "logs"

# Model checkpoints (populated by deploy script)
CHECKPOINTS = {
    "vesicles": MODELS_DIR / "retina_vesicles_best.pt",
    "mitochondria": MODELS_DIR / "retina_mitochondria_best.pt",
    "membrane": MODELS_DIR / "retina_membrane_best.pt",
}

# Model architecture
MODEL_PARAMS = {
    "in_channels": 1,
    "out_channels": 1,
    "depth": 4,
    "initial_features": 32,
    "final_activation": None,  # focal loss models have no built-in activation
}

# Inference
DEFAULT_THRESHOLD = 0.5
TILE_SIZE = 256
TILE_OVERLAP = 32
GPU_ID = 0

# Structures and colors (BGR for OpenCV)
STRUCTURES = {
    "vesicles": {"color": (0, 255, 255), "label": "Synaptic Vesicles", "min_area": 25},
    "mitochondria": {"color": (0, 255, 0), "label": "Mitochondria", "min_area": 5000},
    "membrane": {"color": (255, 0, 0), "label": "Presynaptic Membrane", "min_area": 50000},
}

# Overlay
OVERLAY_ALPHA = 0.35

# Web
APP_HOST = "0.0.0.0"
APP_PORT = 7861
MAX_DISPLAY_SIZE = 2048  # Resize for Gradio display

# Auth
AUTH_PASSWORD = "jablonskilab2026"  # Simple shared password gate
