"""SynapseNet-Retina inference engine.

Matches the combination optimization pipeline from node01 exactly:
- Min-max normalization to [0,1]
- 256px patches, stride 192, zero-padded per patch
- torch.sigmoid for activation (1-channel models), Sigmoid built-in (2-channel)
- Channel 0 = foreground (SynapseNet convention)
- TTA: 7 geometric augmentations (for mitochondria)
- Ensemble: element-wise max of multiple models (for mitochondria)
- Post-processing: morphological opening + min area filtering
- Vesicle ribbon-proximity masking: predictions masked to within N px of ribbon
"""
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt
from unet import UNet2d
from skimage.morphology import disk, opening
from skimage.measure import label as skimage_label
import config

# Cache loaded models: {checkpoint_path_str: (model, device, has_sigmoid)}
_model_cache = {}


def _load_model(checkpoint_path) -> tuple:
    """Load a 2D UNet model from a checkpoint path.

    Auto-detects out_channels and final activation from state dict.
    """
    cache_key = str(checkpoint_path)
    if cache_key in _model_cache:
        return _model_cache[cache_key]

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = torch.device(f"cuda:{config.GPU_ID}" if torch.cuda.is_available() else "cpu")
    save_dict = torch.load(str(checkpoint_path), map_location=device, weights_only=False)

    # Extract state dict from various checkpoint formats
    if isinstance(save_dict, dict) and "model_state" in save_dict:
        state_dict = save_dict["model_state"]
    elif isinstance(save_dict, dict) and "state_dict" in save_dict:
        state_dict = save_dict["state_dict"]
    elif isinstance(save_dict, dict) and any(k.startswith("encoder.") for k in save_dict.keys()):
        state_dict = save_dict
    else:
        state_dict = save_dict.state_dict() if hasattr(save_dict, "state_dict") else save_dict

    # Auto-detect out_channels from out_conv layer
    out_key = "out_conv.weight"
    out_channels = state_dict[out_key].shape[0] if out_key in state_dict else 1

    # 2-channel models (DA) have built-in Sigmoid; 1-channel models (fine-tuned) do not
    has_sigmoid = out_channels == 2
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

    _model_cache[cache_key] = (model, device, has_sigmoid)
    return model, device, has_sigmoid


def _normalize(image_gray):
    """Min-max normalize image to [0,1] — matches training pipeline."""
    img = image_gray.astype(np.float32)
    img_min = img.min()
    img_max = img.max()
    return (img - img_min) / (img_max - img_min + 1e-8)


def _tiled_inference(model, img_norm, device, has_sigmoid=False):
    """Run tiled 2D inference matching node01 evaluation code exactly.

    Uses 256px patches, stride 192, zero-padding per patch.
    Channel 0 = foreground (SynapseNet convention).
    """
    h, w = img_norm.shape
    ps = config.TILE_SIZE
    stride = config.TILE_STRIDE

    pred_fg = np.zeros((h, w), dtype=np.float32)
    count = np.zeros((h, w), dtype=np.float32)

    with torch.no_grad():
        for y in range(0, h, stride):
            for x in range(0, w, stride):
                y1, y2 = y, min(y + ps, h)
                x1, x2 = x, min(x + ps, w)

                # Zero-padded patch (matches notebook exactly)
                patch = np.zeros((ps, ps), dtype=np.float32)
                patch[:y2 - y1, :x2 - x1] = img_norm[y1:y2, x1:x2]

                inp = torch.from_numpy(patch[None, None]).to(device)
                out = model(inp)

                if not has_sigmoid:
                    out = torch.sigmoid(out)

                out = out.cpu().numpy()[0]
                # Channel 0 = foreground (SynapseNet convention)
                pred_fg[y1:y2, x1:x2] += out[0, :y2 - y1, :x2 - x1]
                count[y1:y2, x1:x2] += 1.0

    pred_fg /= (count + 1e-8)
    return pred_fg


def _apply_tta(model, img_norm, device, has_sigmoid=False):
    """Apply test-time augmentation with 7 geometric transforms.

    Transforms: identity, hflip, vflip, hvflip, rot90, rot180, rot270.
    Returns the averaged probability map.
    """
    augmentations = [
        ("identity", lambda x: x, lambda x: x),
        ("hflip", lambda x: np.flip(x, axis=1).copy(), lambda x: np.flip(x, axis=1).copy()),
        ("vflip", lambda x: np.flip(x, axis=0).copy(), lambda x: np.flip(x, axis=0).copy()),
        ("hvflip", lambda x: np.flip(np.flip(x, axis=0), axis=1).copy(),
                   lambda x: np.flip(np.flip(x, axis=0), axis=1).copy()),
        ("rot90", lambda x: np.rot90(x, k=1).copy(), lambda x: np.rot90(x, k=-1).copy()),
        ("rot180", lambda x: np.rot90(x, k=2).copy(), lambda x: np.rot90(x, k=-2).copy()),
        ("rot270", lambda x: np.rot90(x, k=3).copy(), lambda x: np.rot90(x, k=-3).copy()),
    ]

    h, w = img_norm.shape
    prob_sum = np.zeros((h, w), dtype=np.float64)

    for name, forward_aug, inverse_aug in augmentations:
        aug_img = forward_aug(img_norm)
        aug_pred = _tiled_inference(model, aug_img, device, has_sigmoid=has_sigmoid)
        orig_pred = inverse_aug(aug_pred)
        prob_sum += orig_pred

    return (prob_sum / len(augmentations)).astype(np.float32)


def _infer_single_model(checkpoint_path, img_norm, use_tta=False):
    """Run inference with a single model, optionally with TTA."""
    model, device, has_sigmoid = _load_model(checkpoint_path)

    if use_tta:
        return _apply_tta(model, img_norm, device, has_sigmoid=has_sigmoid)
    else:
        return _tiled_inference(model, img_norm, device, has_sigmoid=has_sigmoid)


def postprocess(prob_map, structure, threshold=None, min_area=None):
    """Threshold, morphological cleanup, and instance labeling."""
    if threshold is None:
        threshold = config.STRUCTURES.get(structure, {}).get("threshold", config.DEFAULT_THRESHOLD)
    if min_area is None:
        min_area = config.STRUCTURES[structure]["min_area"]

    binary = prob_map > threshold

    # Morphological opening
    selem = disk(2)
    binary = opening(binary, selem)

    # Connected components and area filtering
    labeled = skimage_label(binary)
    for region_id in range(1, labeled.max() + 1):
        if np.sum(labeled == region_id) < min_area:
            labeled[labeled == region_id] = 0

    final_binary = labeled > 0
    final_labeled = skimage_label(final_binary)
    return final_binary, final_labeled


def _apply_ribbon_proximity_mask(prob_map, ribbon_binary, radius_px):
    """Mask probability map to within radius_px of ribbon foreground.

    Used for vesicle segmentation — vesicles cluster near the synaptic ribbon.
    """
    if not np.any(ribbon_binary):
        return prob_map
    dist = distance_transform_edt(~ribbon_binary)
    proximity_mask = dist <= radius_px
    return prob_map * proximity_mask


def segment(image_gray, structures=None, thresholds=None, min_areas=None,
            vesicle_ribbon_radius=None):
    """Run full segmentation pipeline matching evaluation code on node01.

    For mitochondria: ensemble of 3 models with TTA and element-wise max.
    For membrane: single model, no TTA.
    For vesicles: single model, ribbon-proximity masking (predictions masked to
                  within N px of predicted ribbon).

    Args:
        image_gray: 2D numpy array (uint8 or float), grayscale EM image
        structures: list of structure names, or None for all
        thresholds: dict of {structure: threshold} or None for per-structure defaults
        min_areas: dict of {structure: min_area} or None for per-structure defaults
        vesicle_ribbon_radius: proximity radius in px for vesicle masking, or None for default

    Returns:
        dict of {structure: {"prob_map", "binary", "instances", "n_instances"}}
    """
    if structures is None:
        structures = list(config.STRUCTURES.keys())
    if thresholds is None:
        thresholds = {}
    if min_areas is None:
        min_areas = {}
    if vesicle_ribbon_radius is None:
        vesicle_ribbon_radius = config.VESICLE_RIBBON_RADIUS

    # Min-max normalize to [0,1] — matches training pipeline
    img_norm = _normalize(image_gray)

    # If vesicles are selected, ensure ribbon is also processed (needed for masking)
    need_ribbon_for_vesicles = "vesicles" in structures
    process_order = list(structures)
    if need_ribbon_for_vesicles and "ribbon" not in process_order:
        # Temporarily add ribbon so we can use it for masking
        process_order.insert(0, "ribbon")

    # Ensure ribbon is processed before vesicles
    if "ribbon" in process_order and "vesicles" in process_order:
        process_order.remove("ribbon")
        vesicle_idx = process_order.index("vesicles")
        process_order.insert(vesicle_idx, "ribbon")

    results = {}
    for struct in process_order:
        if struct not in config.CHECKPOINTS:
            continue

        checkpoint_paths = config.CHECKPOINTS[struct]
        use_tta = config.TTA_AUGMENTATIONS.get(struct, False)
        ensemble_method = config.ENSEMBLE_METHOD.get(struct)

        # Filter to only existing checkpoints
        existing_paths = [p for p in checkpoint_paths if p.exists()]
        if not existing_paths:
            continue

        if len(existing_paths) == 1 or ensemble_method is None:
            # Single model inference
            prob_map = _infer_single_model(existing_paths[0], img_norm, use_tta=use_tta)
        else:
            # Ensemble inference
            prob_maps = []
            for ckpt_path in existing_paths:
                pm = _infer_single_model(ckpt_path, img_norm, use_tta=use_tta)
                prob_maps.append(pm)

            if ensemble_method == "max":
                prob_map = np.maximum.reduce(prob_maps)
            elif ensemble_method == "avg":
                prob_map = np.mean(prob_maps, axis=0)
            else:
                prob_map = np.maximum.reduce(prob_maps)

        # For vesicles: apply ribbon-proximity masking
        if struct == "vesicles" and "ribbon" in results:
            ribbon_binary = results["ribbon"]["binary"]
            prob_map = _apply_ribbon_proximity_mask(
                prob_map, ribbon_binary, vesicle_ribbon_radius
            )

        thresh = thresholds.get(struct)
        min_a = min_areas.get(struct)
        binary, instances = postprocess(prob_map, struct, threshold=thresh, min_area=min_a)
        results[struct] = {
            "prob_map": prob_map,
            "binary": binary,
            "instances": instances,
            "n_instances": instances.max(),
        }

    # Remove ribbon from results if it was only added for vesicle masking
    if need_ribbon_for_vesicles and "ribbon" not in structures and "ribbon" in results:
        del results["ribbon"]

    return results
