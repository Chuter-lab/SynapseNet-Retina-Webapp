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


def create_individual_overlay(image_gray, results, struct_name, alpha=None):
    """Create a single-structure overlay on the grayscale image.

    Args:
        image_gray: 2D uint8 grayscale image
        results: dict from inference.segment()
        struct_name: which structure to overlay
        alpha: overlay transparency (0-1)

    Returns:
        3-channel BGR overlay image (uint8)
    """
    return create_overlay(image_gray, results, alpha=alpha, show_structures=[struct_name])


def create_display_panel(image_gray, results, selected):
    """Create a combined side-by-side panel: Input | per-structure overlays.

    Args:
        image_gray: 2D uint8 grayscale image
        results: dict from inference.segment()
        selected: list of selected structure names

    Returns:
        3-channel BGR combined panel image (uint8)
    """
    if image_gray.dtype != np.uint8:
        img_u8 = (np.clip(image_gray, 0, 255)).astype(np.uint8)
    else:
        img_u8 = image_gray

    base = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)
    panels = [base]

    for struct in config.STRUCTURES:
        if struct in selected and struct in results:
            panels.append(create_individual_overlay(image_gray, results, struct))
        else:
            panels.append(base)

    return np.hstack(panels)


def create_panel(image_gray, results):
    """Create binary mask panels for each structure.

    Returns a dict of {name: BGR image} for each panel.
    """
    if image_gray.dtype != np.uint8:
        img_u8 = (np.clip(image_gray, 0, 255)).astype(np.uint8)
    else:
        img_u8 = image_gray

    panels = {}
    for struct_name, struct_result in results.items():
        mask_img = np.zeros_like(img_u8)
        mask_img[struct_result["binary"]] = 255
        panels[struct_name] = cv2.cvtColor(mask_img, cv2.COLOR_GRAY2BGR)

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


def compute_morphometrics(results, image_shape, scale_nm=None):
    """Compute extended morphometric metrics from segmentation results.

    Args:
        results: dict from inference.segment()
        image_shape: (height, width) of the original image
        scale_nm: pixel scale in nm/px, or None for pixel units

    Returns:
        dict keyed by structure name, each containing metric name->value pairs.
    """
    total_pixels = image_shape[0] * image_shape[1]
    # Scale factor: convert px^2 to physical area, px to physical length
    area_factor = (scale_nm ** 2) if scale_nm else 1.0
    len_factor = scale_nm if scale_nm else 1.0
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
            m["mean_area"] = round(float(np.mean(areas)) * area_factor, 2)
            m["min_area"] = round(float(np.min(areas)) * area_factor, 2)
            m["max_area"] = round(float(np.max(areas)) * area_factor, 2)
            m["std_area"] = round(float(np.std(areas)) * area_factor, 2)

            # Circularity and aspect ratio for mitochondria, ribbon, and vesicles
            if struct_name in ("mitochondria", "ribbon", "vesicles"):
                circularities = []
                aspect_ratios = []
                major_lengths = []
                for p in props:
                    if p.perimeter > 0:
                        circ = 4 * np.pi * p.area / (p.perimeter ** 2)
                        circularities.append(circ)
                    minor = getattr(p, "axis_minor_length", None) or getattr(p, "minor_axis_length", 0)
                    major = getattr(p, "axis_major_length", None) or getattr(p, "major_axis_length", 0)
                    if minor > 0:
                        aspect_ratios.append(major / minor)
                    if major > 0:
                        major_lengths.append(major)
                if circularities:
                    m["mean_circularity"] = round(float(np.mean(circularities)), 4)
                if aspect_ratios:
                    m["mean_aspect_ratio"] = round(float(np.mean(aspect_ratios)), 4)
                if struct_name in ("ribbon", "vesicles") and major_lengths:
                    m["mean_length"] = round(float(np.mean(major_lengths)) * len_factor, 2)

            # Membrane-specific: total perimeter
            if struct_name == "membrane":
                perimeters = [p.perimeter for p in props]
                m["total_perimeter"] = round(float(np.sum(perimeters)) * len_factor, 2)

        else:
            m["mean_area"] = 0.0
            m["min_area"] = 0
            m["max_area"] = 0
            m["std_area"] = 0.0

        metrics[struct_name] = m

    return metrics


def format_morphometrics_markdown(metrics):
    """Format morphometric metrics as markdown tables.

    Args:
        metrics: dict from compute_morphometrics()

    Returns:
        Markdown string with per-structure tables
    """
    sections = []

    for struct_name in config.STRUCTURES:
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
        if "mean_length" in m:
            rows.append(f"| Mean length (px) | {m['mean_length']:.1f} |")
        if "total_perimeter" in m:
            rows.append(f"| Total perimeter (px) | {m['total_perimeter']:.1f} |")

        sections.append("\n".join(rows))

    return "\n\n".join(sections)


def format_morphometrics_html(metrics, scale_nm=None):
    """Format morphometric metrics as side-by-side HTML tables.

    Args:
        metrics: dict from compute_morphometrics()
        scale_nm: pixel scale in nm/px, or None for pixel units

    Returns:
        HTML string with side-by-side per-structure tables
    """
    # Determine unit labels
    if scale_nm:
        area_unit = "nm\u00B2"
        len_unit = "nm"
    else:
        area_unit = "px"
        len_unit = "px"

    table_style = (
        "border-collapse:collapse; width:100%; font-size:0.9em; font-family:'Space Mono',monospace;"
    )
    th_style = (
        "text-align:left; padding:6px 10px; border-bottom:2px solid #cbd5e1; "
        "color:#334155; font-weight:600;"
    )
    td_style = "padding:5px 10px; border-bottom:1px solid #e2e8f0;"
    td_val_style = td_style + " text-align:right; font-variant-numeric:tabular-nums;"

    panels = []
    for struct_name in config.STRUCTURES:
        if struct_name not in metrics:
            continue
        m = metrics[struct_name]
        label = config.STRUCTURES[struct_name]["label"]
        color_bgr = config.STRUCTURES[struct_name]["color"]
        color_hex = "#{:02x}{:02x}{:02x}".format(color_bgr[2], color_bgr[1], color_bgr[0])

        rows_html = ""
        row_data = [
            ("Instances", f"{m['count']}"),
            ("Total area (px)", f"{m['total_area_px']:,}"),
            ("Coverage (%)", f"{m['coverage_pct']:.2f}"),
            (f"Mean area ({area_unit})", f"{m['mean_area']:,.1f}"),
            (f"Min area ({area_unit})", f"{m['min_area']:,.1f}"),
            (f"Max area ({area_unit})", f"{m['max_area']:,.1f}"),
            (f"Std area ({area_unit})", f"{m['std_area']:,.1f}"),
        ]
        if "mean_circularity" in m:
            row_data.append(("Mean circularity", f"{m['mean_circularity']:.4f}"))
        if "mean_aspect_ratio" in m:
            row_data.append(("Mean aspect ratio", f"{m['mean_aspect_ratio']:.4f}"))
        if "mean_length" in m:
            row_data.append((f"Mean length ({len_unit})", f"{m['mean_length']:,.1f}"))
        if "total_perimeter" in m:
            row_data.append((f"Total perimeter ({len_unit})", f"{m['total_perimeter']:,.1f}"))

        for metric, value in row_data:
            rows_html += f'<tr><td style="{td_style}">{metric}</td><td style="{td_val_style}">{value}</td></tr>'

        panel = f'''
        <div style="flex:1; min-width:280px;">
            <div style="font-weight:700; font-size:1em; margin-bottom:6px; color:{color_hex};">
                {label}
            </div>
            <table style="{table_style}">
                <thead><tr>
                    <th style="{th_style}">Metric</th>
                    <th style="{th_style} text-align:right;">Value</th>
                </tr></thead>
                <tbody>{rows_html}</tbody>
            </table>
        </div>'''
        panels.append(panel)

    return f'<div style="display:flex; gap:24px; flex-wrap:wrap;">{"".join(panels)}</div>'


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

        for struct_name in config.STRUCTURES:
            if struct_name not in metrics:
                continue
            label = config.STRUCTURES[struct_name]["label"]
            for metric_name, value in metrics[struct_name].items():
                writer.writerow([label, metric_name, value])

    return output_path
