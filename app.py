"""THIRU — TEM Histological Image Recognition for Ultrastructure.

Gradio-based interface for retinal EM synapse segmentation.
Segments mitochondria and presynaptic membrane.
"""
import gradio as gr
import numpy as np
import cv2
import tifffile
import time
import zipfile
from pathlib import Path
from PIL import Image
import torch

import config
import inference
import visualization

STATIC_DIR = Path(__file__).parent / "static"


def _ensure_dirs():
    """Create required directories."""
    for d in [config.UPLOAD_DIR, config.TEMP_DIR, config.LOG_DIR, config.MODELS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def _load_image(file_path):
    """Load image from file path, return grayscale numpy array."""
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix in (".tif", ".tiff"):
        img = tifffile.imread(str(path))
    else:
        img = np.array(Image.open(str(path)))

    # Convert to grayscale if needed
    if img.ndim == 3:
        if img.shape[2] == 4:  # RGBA
            img = img[:, :, :3]
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    return img


def _check_models():
    """Check which models are available."""
    available = {}
    for struct, path in config.CHECKPOINTS.items():
        available[struct] = path.exists()
    return available


def _gpu_status_html():
    """Generate GPU status HTML."""
    try:
        if not torch.cuda.is_available():
            return '<span style="color:#e74c3c;">No GPU available</span>'
        name = torch.cuda.get_device_name(config.GPU_ID)
        mem_used = torch.cuda.memory_allocated(config.GPU_ID) / 1024**2
        mem_total = torch.cuda.get_device_properties(config.GPU_ID).total_mem / 1024**2
        return f'<span style="color:#2ecc71;">{name} ({mem_used:.0f}/{mem_total:.0f} MB)</span>'
    except Exception:
        return '<span style="color:#95a5a6;">GPU status unavailable</span>'


def process_image(file, structures, threshold, progress=gr.Progress()):
    """Main processing function called by Gradio."""
    if file is None:
        raise gr.Error("Please upload an image first.")

    t0 = time.time()

    # Parse selected structures
    struct_map = {
        "Mitochondria": "mitochondria",
        "Presynaptic Membrane": "membrane",
    }
    selected = [struct_map[s] for s in structures if s in struct_map]

    if not selected:
        raise gr.Error("Please select at least one structure to segment.")

    # Check model availability
    available = _check_models()
    missing = [s for s in selected if not available.get(s, False)]
    if missing:
        raise gr.Error(f"Models not found for: {', '.join(missing)}. Check deployment.")

    progress(0.1, desc="Loading image...")

    # Load image
    img_gray = _load_image(file)
    h, w = img_gray.shape
    progress(0.2, desc=f"Image loaded: {w}x{h}")

    # Run segmentation (threshold=0 means use per-structure optimal)
    progress(0.3, desc="Running segmentation...")
    seg_threshold = None if threshold == 0.0 else threshold
    results = inference.segment(img_gray, structures=selected, threshold=seg_threshold)

    progress(0.8, desc="Creating visualization...")

    # Prepare grayscale input display
    if img_gray.dtype != np.uint8:
        img_u8 = (np.clip(img_gray, 0, 255)).astype(np.uint8)
    else:
        img_u8 = img_gray
    input_rgb = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2RGB)
    input_display = visualization.resize_for_display(input_rgb)

    # Create per-structure overlay images
    mito_overlay_display = None
    membrane_overlay_display = None
    if "mitochondria" in selected:
        mito_bgr = visualization.create_individual_overlay(img_gray, results, "mitochondria")
        mito_overlay_display = visualization.resize_for_display(
            cv2.cvtColor(mito_bgr, cv2.COLOR_BGR2RGB)
        )
    if "membrane" in selected:
        mem_bgr = visualization.create_individual_overlay(img_gray, results, "membrane")
        membrane_overlay_display = visualization.resize_for_display(
            cv2.cvtColor(mem_bgr, cv2.COLOR_BGR2RGB)
        )

    # Compute morphometric metrics
    metrics = visualization.compute_morphometrics(results, img_gray.shape)
    elapsed = time.time() - t0
    metrics_text = visualization.format_morphometrics_markdown(metrics)
    metrics_text += f"\n\n**Processing time:** {elapsed:.1f}s | **Image size:** {w}x{h}"

    # Build download files — descriptive names, TIF format for images
    download_files = []
    zip_contents = []  # (path, arcname) for zip

    # 1. Combined display panel
    panel_bgr = visualization.create_display_panel(img_gray, results, selected)
    panel_path = config.TEMP_DIR / "THIRU_display_panel.tif"
    tifffile.imwrite(str(panel_path), cv2.cvtColor(panel_bgr, cv2.COLOR_BGR2RGB))
    download_files.append(str(panel_path))
    zip_contents.append(str(panel_path))

    # 2. Individual overlay TIFs
    for struct in selected:
        label = config.STRUCTURES[struct]["label"].replace(" ", "_")
        ov_bgr = visualization.create_individual_overlay(img_gray, results, struct)
        ov_path = config.TEMP_DIR / f"THIRU_overlay_{label}.tif"
        tifffile.imwrite(str(ov_path), cv2.cvtColor(ov_bgr, cv2.COLOR_BGR2RGB))
        download_files.append(str(ov_path))
        zip_contents.append(str(ov_path))

    # 3. Individual binary mask TIFs
    for struct in selected:
        label = config.STRUCTURES[struct]["label"].replace(" ", "_")
        mask_path = config.TEMP_DIR / f"THIRU_mask_{label}.tif"
        mask_u8 = (results[struct]["binary"].astype(np.uint8) * 255)
        tifffile.imwrite(str(mask_path), mask_u8)
        download_files.append(str(mask_path))
        zip_contents.append(str(mask_path))

    # 4. Metrics CSV
    csv_path = config.TEMP_DIR / "THIRU_metrics.csv"
    visualization.export_metrics_csv(metrics, csv_path)
    download_files.append(str(csv_path))
    zip_contents.append(str(csv_path))

    # 5. Download All zip
    zip_path = config.TEMP_DIR / "THIRU_results.zip"
    with zipfile.ZipFile(str(zip_path), "w", zipfile.ZIP_DEFLATED) as zf:
        for fpath in zip_contents:
            zf.write(fpath, Path(fpath).name)
    download_files.insert(0, str(zip_path))  # zip first in list

    progress(1.0, desc="Done!")

    return (
        input_display,
        mito_overlay_display,
        membrane_overlay_display,
        metrics_text,
        download_files,
    )


def _load_logo_svg():
    """Load the THIRU logo SVG for inline embedding."""
    svg_path = STATIC_DIR / "thiru_logo.svg"
    if svg_path.exists():
        return svg_path.read_text()
    return None


def create_app():
    """Build the Gradio application."""
    _ensure_dirs()

    available = _check_models()
    model_labels = [config.STRUCTURES[s]["label"] for s, ok in available.items() if ok]
    model_status = " | ".join(model_labels)

    logo_svg = _load_logo_svg()

    css = """
    :root {
        --color-accent: #2563eb;
        --color-accent-soft: #dbeafe;
    }
    .model-status { font-size: 0.9em; padding: 8px 12px; background: #f8fafc; border-radius: 6px; }
    .thiru-header { margin-bottom: 8px; }
    .thiru-header svg { max-height: 80px; width: auto; }
    .thiru-subtitle { color: #64748b; font-size: 0.95em; margin-top: 4px; letter-spacing: 0.5px; }
    .thiru-policy { color: #94a3b8; font-size: 0.8em; line-height: 1.5; margin-top: 10px; }
    .thiru-policy p { margin: 4px 0; }
    .thiru-policy strong { color: #64748b; }
    footer { display: none !important; }
    /* Hide Gradio PWA install banner */
    .pwa-install-container, .pwa-toast, [class*="pwa"] { display: none !important; }
    """

    with gr.Blocks(title="THIRU") as app:
        # Store css/theme for launch() (Gradio 6 moved these from Blocks to launch)
        app._thiru_css = css
        app._thiru_theme = gr.themes.Soft(primary_hue="blue")
        if logo_svg:
            gr.HTML(f"""
            <div class="thiru-header">
                {logo_svg}
                <div class="thiru-subtitle">TEM Histological Image Recognition for Ultrastructure</div>
                <div class="thiru-policy">
                    <p><strong>Data policy:</strong> Uploaded images are stored temporarily and deleted within 24 hours. Cached results are retained for up to 30 days, then automatically removed.</p>
                    <p><strong>License:</strong> This software is licensed under CC BY-NC-ND 4.0. Free for academic and non-commercial use with attribution. Commercial use and derivative works are prohibited.</p>
                </div>
            </div>
            """)
        else:
            gr.HTML("""
            <div class="thiru-header">
                <h1 style="margin:0;">THIRU</h1>
                <div class="thiru-subtitle">TEM Histological Image Recognition for Ultrastructure</div>
                <div class="thiru-policy">
                    <p><strong>Data policy:</strong> Uploaded images are stored temporarily and deleted within 24 hours. Cached results are retained for up to 30 days, then automatically removed.</p>
                    <p><strong>License:</strong> This software is licensed under CC BY-NC-ND 4.0. Free for academic and non-commercial use with attribution. Commercial use and derivative works are prohibited.</p>
                </div>
            </div>
            """)

        gr.HTML(f'<div class="model-status"><b>Models:</b> {model_status}</div>')

        with gr.Row():
            # Left column: inputs
            with gr.Column(scale=1):
                file_input = gr.File(
                    label="Upload EM Image",
                    file_types=[".tif", ".tiff", ".png", ".jpg"],
                    type="filepath",
                )

                structures_input = gr.CheckboxGroup(
                    choices=["Mitochondria", "Presynaptic Membrane"],
                    value=["Mitochondria", "Presynaptic Membrane"],
                    label="Structures to Segment",
                )

                threshold_input = gr.Slider(
                    minimum=0.0,
                    maximum=0.9,
                    value=0.0,
                    step=0.05,
                    label="Prediction Threshold",
                    info="0 = use per-structure optimal thresholds (recommended). Otherwise: lower = more sensitive, higher = more specific.",
                )

                run_btn = gr.Button("Run Segmentation", variant="primary", size="lg")

                gr.HTML(f'<div style="margin-top:12px;">{_gpu_status_html()}</div>')

            # Right column: results
            with gr.Column(scale=2):
                # Three-panel display: Input | Mito overlay | Membrane overlay
                with gr.Row():
                    input_image = gr.Image(
                        label="Input",
                        interactive=False,
                    )
                    mito_overlay = gr.Image(
                        label="Mitochondria",
                        interactive=False,
                    )
                    membrane_overlay = gr.Image(
                        label="Membrane",
                        interactive=False,
                    )

                # Morphometric analysis
                results_md = gr.Markdown(label="Morphometric Analysis")

                # Downloads
                download_files = gr.File(
                    label="Download Results",
                    file_count="multiple",
                    interactive=False,
                )

        # Wire up the run button
        run_btn.click(
            fn=process_image,
            inputs=[file_input, structures_input, threshold_input],
            outputs=[
                input_image,
                mito_overlay,
                membrane_overlay,
                results_md,
                download_files,
            ],
        )

        gr.HTML("""
        <div style="text-align:center; padding:16px 0 8px; margin-top:24px;
                    border-top:1px solid #e2e8f0; color:#94a3b8; font-size:0.85em;">
            Lead Developer: Benton Chuter, MD, MS
        </div>
        """)

        # Replace Gradio's favicon and remove PWA manifest after page loads
        app.load(fn=None, js=_onload_js)

    return app


# JS to replace Gradio's default favicon and remove PWA manifest
_onload_js = """
() => {
    function setFavicon() {
        document.querySelectorAll('link[rel*="icon"]').forEach(el => el.remove());
        const link = document.createElement('link');
        link.rel = 'icon';
        link.type = 'image/svg+xml';
        link.href = '/favicon.svg?v=3';
        document.head.appendChild(link);
    }
    function removeManifest() {
        document.querySelectorAll('link[rel="manifest"]').forEach(el => el.remove());
    }
    setFavicon();
    removeManifest();
    // Watch for Gradio re-injecting its favicon or manifest
    new MutationObserver(() => {
        const badFav = document.querySelector('link[rel*="icon"]:not([href*="/favicon.svg"])');
        const manifest = document.querySelector('link[rel="manifest"]');
        if (badFav || manifest) { setFavicon(); removeManifest(); }
    }).observe(document.head, { childList: true });
}
"""


def main():
    from starlette.responses import JSONResponse, Response
    from starlette.routing import Route

    app = create_app()

    favicon_svg_path = STATIC_DIR / "favicon.svg"
    favicon_svg_bytes = favicon_svg_path.read_bytes() if favicon_svg_path.exists() else b""
    favicon_ico_path = STATIC_DIR / "favicon.ico"
    favicon_ico_bytes = favicon_ico_path.read_bytes() if favicon_ico_path.exists() else b""

    # Launch (non-blocking so we can patch routes after)
    launch_kwargs = dict(
        server_name=config.APP_HOST,
        server_port=config.APP_PORT,
        share=False,
        show_error=True,
        auth=config.AUTH_CREDENTIALS,
        auth_message="THIRU — TEM Histological Image Recognition for Ultrastructure. Enter credentials to access.",
        pwa=False,
        prevent_thread_lock=True,
        allowed_paths=[str(config.TEMP_DIR), str(config.UPLOAD_DIR)],
    )
    # Gradio 6 accepts css/theme in launch()
    if hasattr(app, "_thiru_css"):
        launch_kwargs["css"] = app._thiru_css
    if hasattr(app, "_thiru_theme"):
        launch_kwargs["theme"] = app._thiru_theme
    app.launch(**launch_kwargs)

    # Override Gradio's manifest.json (disables "Open in app")
    async def empty_manifest(request):
        return JSONResponse({})

    # Override Gradio's favicon with our SVG (preserves gradients/colors)
    async def serve_favicon(request):
        return Response(
            content=favicon_svg_bytes,
            media_type="image/svg+xml",
            headers={"Cache-Control": "no-cache, must-revalidate", "CDN-Cache-Control": "no-store"},
        )

    # Also serve ICO fallback for older browsers
    async def serve_favicon_ico(request):
        return Response(
            content=favicon_ico_bytes,
            media_type="image/x-icon",
            headers={"Cache-Control": "no-cache, must-revalidate", "CDN-Cache-Control": "no-store"},
        )

    app.app.router.routes.insert(0, Route("/manifest.json", empty_manifest))
    app.app.router.routes.insert(0, Route("/favicon.svg", serve_favicon))
    app.app.router.routes.insert(0, Route("/favicon.ico", serve_favicon_ico))

    # Block the main thread
    import threading
    threading.Event().wait()


if __name__ == "__main__":
    main()
