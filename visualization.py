"""Visualization utilities for SynapseNet-Retina webapp."""
import numpy as np
import cv2
import csv
from skimage.measure import regionprops
import config


def create_overlay(image_gray, results, alpha=None, show_structures=None):
    """Create color overlay of segmentation results on grayscale image.

    Args:
        image_gray: 2D uint8 grayscale image
        results: dict from inference.segment()
        alpha: overlay transparency (0-1)
        show_structures: list of structure names to draw, or None for all

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
        if show_structures is not None and struct_name not in show_structures:
            continue
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


def compute_morphometrics(results, image_shape):
    """Compute extended morphometric metrics from segmentation results.

    Args:
        results: dict from inference.segment()
        image_shape: (height, width) of the original image

    Returns:
        dict keyed by structure name, each containing metric name→value pairs.
        Also includes a "_cross" key for cross-structure metrics.
    """
    total_pixels = image_shape[0] * image_shape[1]
    metrics = {}

    for struct_name, struct_result in results.items():
        instances = struct_result["instances"]
        binary = struct_result["binary"]
        n_inst = struct_result["n_instances"]

        m = {}
        m["count"] = int(n_inst)
        m["total_area_px"] = int(np.sum(binary))
        m["coverage_pct"] = round(100.0 * m["total_area_px"] / total_pixels, 4) if total_pixels > 0 else 0.0

        if n_inst > 0:
            props = regionprops(instances)
            areas = np.array([p.area for p in props])
            m["mean_area"] = round(float(np.mean(areas)), 2)
            m["min_area"] = int(np.min(areas))
            m["max_area"] = int(np.max(areas))
            m["std_area"] = round(float(np.std(areas)), 2)

            # Circularity and aspect ratio for vesicles and mitochondria
            if struct_name in ("vesicles", "mitochondria"):
                circularities = []
                aspect_ratios = []
                for p in props:
                    if p.perimeter > 0:
                        circ = 4 * np.pi * p.area / (p.perimeter ** 2)
                        circularities.append(circ)
                    minor = getattr(p, "axis_minor_length", None) or getattr(p, "minor_axis_length", 0)
                    major = getattr(p, "axis_major_length", None) or getattr(p, "major_axis_length", 0)
                    if minor > 0:
                        aspect_ratios.append(major / minor)
                if circularities:
                    m["mean_circularity"] = round(float(np.mean(circularities)), 4)
                if aspect_ratios:
                    m["mean_aspect_ratio"] = round(float(np.mean(aspect_ratios)), 4)

            # Membrane-specific: total perimeter
            if struct_name == "membrane":
                perimeters = [p.perimeter for p in props]
                m["total_perimeter"] = round(float(np.sum(perimeters)), 2)

            # Vesicle density (count per megapixel)
            if struct_name == "vesicles":
                mpx = total_pixels / 1e6
                m["density_per_mpx"] = round(n_inst / mpx, 2) if mpx > 0 else 0.0
        else:
            m["mean_area"] = 0.0
            m["min_area"] = 0
            m["max_area"] = 0
            m["std_area"] = 0.0

        metrics[struct_name] = m

    # Cross-structure metrics
    cross = {}
    if "vesicles" in results and "membrane" in results:
        membrane_mask = results["membrane"]["binary"]
        vesicle_instances = results["vesicles"]["instances"]

        # Count vesicles within membrane-defined terminal
        if np.any(membrane_mask) and vesicle_instances.max() > 0:
            # Find vesicle labels that overlap with membrane region
            vesicle_labels_in_terminal = np.unique(vesicle_instances[membrane_mask])
            vesicle_labels_in_terminal = vesicle_labels_in_terminal[vesicle_labels_in_terminal > 0]
            cross["vesicles_in_terminal"] = int(len(vesicle_labels_in_terminal))

            # Density within terminal
            terminal_area = int(np.sum(membrane_mask))
            if terminal_area > 0:
                terminal_mpx = terminal_area / 1e6
                cross["vesicle_density_in_terminal"] = round(
                    cross["vesicles_in_terminal"] / terminal_mpx, 2
                ) if terminal_mpx > 0 else 0.0

    if cross:
        metrics["_cross"] = cross

    return metrics


def format_morphometrics_markdown(metrics):
    """Format morphometric metrics as markdown tables.

    Args:
        metrics: dict from compute_morphometrics()

    Returns:
        Markdown string with per-structure tables
    """
    sections = []

    for struct_name in ("vesicles", "mitochondria", "membrane"):
        if struct_name not in metrics:
            continue
        m = metrics[struct_name]
        label = config.STRUCTURES[struct_name]["label"]

        rows = []
        rows.append(f"**{label}**\n")
        rows.append("| Metric | Value |")
        rows.append("|--------|-------|")
        rows.append(f"| Instances | {m['count']} |")
        rows.append(f"| Total area (px) | {m['total_area_px']:,} |")
        rows.append(f"| Coverage (%) | {m['coverage_pct']:.2f} |")
        rows.append(f"| Mean area (px) | {m['mean_area']:.1f} |")
        rows.append(f"| Min area (px) | {m['min_area']:,} |")
        rows.append(f"| Max area (px) | {m['max_area']:,} |")
        rows.append(f"| Std area (px) | {m['std_area']:.1f} |")

        if "mean_circularity" in m:
            rows.append(f"| Mean circularity | {m['mean_circularity']:.4f} |")
        if "mean_aspect_ratio" in m:
            rows.append(f"| Mean aspect ratio | {m['mean_aspect_ratio']:.4f} |")
        if "total_perimeter" in m:
            rows.append(f"| Total perimeter (px) | {m['total_perimeter']:.1f} |")
        if "density_per_mpx" in m:
            rows.append(f"| Density (per Mpx) | {m['density_per_mpx']:.1f} |")

        sections.append("\n".join(rows))

    # Cross-structure metrics
    if "_cross" in metrics:
        cross = metrics["_cross"]
        rows = []
        rows.append("**Cross-structure**\n")
        rows.append("| Metric | Value |")
        rows.append("|--------|-------|")
        if "vesicles_in_terminal" in cross:
            rows.append(f"| Vesicles in terminal | {cross['vesicles_in_terminal']} |")
        if "vesicle_density_in_terminal" in cross:
            rows.append(f"| Vesicle density in terminal (per Mpx) | {cross['vesicle_density_in_terminal']:.1f} |")
        sections.append("\n".join(rows))

    return "\n\n".join(sections)


def export_metrics_csv(metrics, output_path):
    """Write morphometric metrics to CSV file.

    Args:
        metrics: dict from compute_morphometrics()
        output_path: path to write CSV

    Returns:
        str path to written file
    """
    output_path = str(output_path)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Structure", "Metric", "Value"])

        for struct_name in ("vesicles", "mitochondria", "membrane"):
            if struct_name not in metrics:
                continue
            label = config.STRUCTURES[struct_name]["label"]
            for metric_name, value in metrics[struct_name].items():
                writer.writerow([label, metric_name, value])

        if "_cross" in metrics:
            for metric_name, value in metrics["_cross"].items():
                writer.writerow(["Cross-structure", metric_name, value])

    return output_path


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
