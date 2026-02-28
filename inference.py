"""SynapseNet-Retina inference engine with tiled prediction and caching."""
import numpy as np
import torch
import torch.nn.functional as F
from torch_em.model import UNet2d
from pathlib import Path
from skimage.morphology import disk, binary_opening
from skimage.measure import label as skimage_label
import config

# Cache loaded models
_model_cache = {}


def _load_model(structure: str) -> tuple:
    """Load and cache a UNet2d model for the given structure."""
    if structure in _model_cache:
        return _model_cache[structure]

    ckpt_path = config.CHECKPOINTS[structure]
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    device = torch.device(f"cuda:{config.GPU_ID}" if torch.cuda.is_available() else "cpu")
    save_dict = torch.load(str(ckpt_path), map_location=device, weights_only=False)

    if "model_state" in save_dict:
        state_dict = save_dict["model_state"]
    elif isinstance(save_dict, dict) and "state_dict" in save_dict:
        state_dict = save_dict["state_dict"]
    else:
        state_dict = save_dict.state_dict() if hasattr(save_dict, "state_dict") else save_dict

    # Detect out_channels from state dict
    out_key = "out_conv.weight"
    out_channels = state_dict[out_key].shape[0] if out_key in state_dict else config.MODEL_PARAMS["out_channels"]
    has_sigmoid = out_channels == 2  # pretrained models with 2 channels have sigmoid

    final_act = "Sigmoid" if has_sigmoid else config.MODEL_PARAMS["final_activation"]

    model = UNet2d(
        in_channels=config.MODEL_PARAMS["in_channels"],
        out_channels=out_channels,
        depth=config.MODEL_PARAMS["depth"],
        initial_features=config.MODEL_PARAMS["initial_features"],
        final_activation=final_act,
    )
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    _model_cache[structure] = (model, device, has_sigmoid, out_channels)
    return model, device, has_sigmoid, out_channels


def tiled_inference(model, img_norm, device, patch_size=256, has_sigmoid=False, out_channels=1):
    """Run tiled inference on a 2D image, handling arbitrary sizes."""
    h, w = img_norm.shape
    overlap = config.TILE_OVERLAP
    stride = patch_size - overlap

    # Pad image to cover full tiles
    pad_h = (stride - (h % stride)) % stride + overlap
    pad_w = (stride - (w % stride)) % stride + overlap
    padded = np.pad(img_norm, ((0, pad_h), (0, pad_w)), mode="reflect")

    # Accumulator for predictions and weight map
    pred_sum = np.zeros((padded.shape[0], padded.shape[1]), dtype=np.float64)
    weight_sum = np.zeros_like(pred_sum)

    ph, pw = padded.shape
    with torch.no_grad():
        for y in range(0, ph - patch_size + 1, stride):
            for x in range(0, pw - patch_size + 1, stride):
                patch = padded[y : y + patch_size, x : x + patch_size]
                tensor = torch.from_numpy(patch).float().unsqueeze(0).unsqueeze(0).to(device)
                out = model(tensor)
                out = out.cpu().numpy()[0, 0]  # channel 0 = foreground

                if not has_sigmoid:
                    out = 1.0 / (1.0 + np.exp(-np.clip(out, -20, 20)))

                pred_sum[y : y + patch_size, x : x + patch_size] += out
                weight_sum[y : y + patch_size, x : x + patch_size] += 1.0

    weight_sum = np.maximum(weight_sum, 1.0)
    pred = (pred_sum / weight_sum)[:h, :w]
    return pred.astype(np.float32)


def postprocess(prob_map, structure, threshold=None):
    """Threshold, morphological cleanup, and instance labeling."""
    if threshold is None:
        threshold = config.DEFAULT_THRESHOLD

    binary = prob_map > threshold
    struct_cfg = config.STRUCTURES[structure]
    min_area = struct_cfg["min_area"]

    # Morphological opening
    selem = disk(2)
    binary = binary_opening(binary, selem)

    # Connected components and area filtering
    labeled = skimage_label(binary)
    for region_id in range(1, labeled.max() + 1):
        if np.sum(labeled == region_id) < min_area:
            labeled[labeled == region_id] = 0

    # Re-label sequentially
    final_binary = labeled > 0
    final_labeled = skimage_label(final_binary)
    return final_binary, final_labeled


def segment(image_gray, structures=None, threshold=None):
    """Run full segmentation pipeline.

    Args:
        image_gray: 2D numpy array (uint8 or float), grayscale EM image
        structures: list of structure names, or None for all
        threshold: prediction threshold (0-1)

    Returns:
        dict of {structure: {"prob_map", "binary", "instances", "n_instances"}}
    """
    if structures is None:
        structures = list(config.STRUCTURES.keys())

    # Normalize to 0-1 float
    if image_gray.dtype == np.uint8:
        img_norm = image_gray.astype(np.float32) / 255.0
    else:
        img_norm = image_gray.astype(np.float32)
        if img_norm.max() > 1.0:
            img_norm = img_norm / img_norm.max()

    results = {}
    for struct in structures:
        if struct not in config.CHECKPOINTS:
            continue

        model, device, has_sigmoid, out_channels = _load_model(struct)
        prob_map = tiled_inference(
            model, img_norm, device,
            patch_size=config.TILE_SIZE,
            has_sigmoid=has_sigmoid,
            out_channels=out_channels,
        )
        binary, instances = postprocess(prob_map, struct, threshold)
        results[struct] = {
            "prob_map": prob_map,
            "binary": binary,
            "instances": instances,
            "n_instances": instances.max(),
        }

    return results
