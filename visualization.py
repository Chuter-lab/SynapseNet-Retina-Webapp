"""Visualization utilities for SynapseNet-Retina webapp."""
import numpy as np
import cv2
import config


def create_overlay(image_gray, results, alpha=None):
    """Create color overlay of segmentation results on grayscale image.

    Args:
        image_gray: 2D uint8 grayscale image
        results: dict from inference.segment()
        alpha: overlay transparency (0-1)

    Returns:
        3-channel BGR overlay image (uint8)
    """
    if alpha is None:
        alpha = config.OVERLAY_ALPHA

    if image_gray.dtype != np.uint8:
        img_u8 = (np.clip(image_gray, 0, 255)).astype(np.uint8)
    else:
        img_u8 = image_gray

    base = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)
    overlay = base.copy()

    for struct_name, struct_result in results.items():
        color = config.STRUCTURES[struct_name]["color"]
        mask = struct_result["binary"]
        overlay[mask] = color

    blended = cv2.addWeighted(overlay, alpha, base, 1 - alpha, 0)
    return blended


def create_panel(image_gray, results):
    """Create a multi-panel visualization.

    Returns a dict of {name: BGR image} for each panel.
    """
    if image_gray.dtype != np.uint8:
        img_u8 = (np.clip(image_gray, 0, 255)).astype(np.uint8)
    else:
        img_u8 = image_gray

    panels = {}
    panels["original"] = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)
    panels["overlay"] = create_overlay(image_gray, results)

    for struct_name, struct_result in results.items():
        # Binary mask panel
        mask_img = np.zeros_like(img_u8)
        mask_img[struct_result["binary"]] = 255
        panels[struct_name] = cv2.cvtColor(mask_img, cv2.COLOR_GRAY2BGR)

        # Probability map panel
        prob = struct_result["prob_map"]
        prob_u8 = (np.clip(prob, 0, 1) * 255).astype(np.uint8)
        prob_color = cv2.applyColorMap(prob_u8, cv2.COLORMAP_JET)
        panels[f"{struct_name}_prob"] = prob_color

    return panels


def resize_for_display(image, max_size=None):
    """Resize image for Gradio display if too large."""
    if max_size is None:
        max_size = config.MAX_DISPLAY_SIZE

    h, w = image.shape[:2]
    if max(h, w) <= max_size:
        return image

    scale = max_size / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def format_results_table(results):
    """Format segmentation results as a markdown table string."""
    rows = []
    rows.append("| Structure | Instances | Pixels | Coverage (%) |")
    rows.append("|-----------|-----------|--------|-------------|")

    for struct_name, struct_result in results.items():
        label = config.STRUCTURES[struct_name]["label"]
        n_inst = struct_result["n_instances"]
        n_pixels = int(np.sum(struct_result["binary"]))
        total = struct_result["binary"].size
        coverage = 100.0 * n_pixels / total if total > 0 else 0.0
        rows.append(f"| {label} | {n_inst} | {n_pixels:,} | {coverage:.2f} |")

    return "\n".join(rows)
