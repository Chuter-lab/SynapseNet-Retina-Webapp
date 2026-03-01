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
import psutil
import threading
import logging
from pathlib import Path
from PIL import Image
import torch

import config
import inference
import visualization

STATIC_DIR = Path(__file__).parent / "static"
log = logging.getLogger("thiru")

# Cleanup intervals
UPLOAD_MAX_AGE_HOURS = 24
RESULT_MAX_AGE_DAYS = 30
CLEANUP_INTERVAL_HOURS = 1


def _ensure_dirs():
    """Create required directories."""
    for d in [config.UPLOAD_DIR, config.TEMP_DIR, config.LOG_DIR, config.MODELS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def _cleanup_old_files():
    """Delete uploads older than 24h and temp results older than 30 days."""
    now = time.time()

    # Uploads: delete after 24 hours
    if config.UPLOAD_DIR.exists():
        for f in config.UPLOAD_DIR.iterdir():
            if f.is_file():
                age_hours = (now - f.stat().st_mtime) / 3600
                if age_hours > UPLOAD_MAX_AGE_HOURS:
                    try:
                        f.unlink()
                        log.info("Cleaned upload: %s (%.1fh old)", f.name, age_hours)
                    except OSError:
                        pass

    # Temp results: delete after 30 days (but not uploads subdir)
    if config.TEMP_DIR.exists():
        for f in config.TEMP_DIR.iterdir():
            if f.is_file():
                age_days = (now - f.stat().st_mtime) / 86400
                if age_days > RESULT_MAX_AGE_DAYS:
                    try:
                        f.unlink()
                        log.info("Cleaned result: %s (%.0fd old)", f.name, age_days)
                    except OSError:
                        pass


def _start_cleanup_thread():
    """Run periodic cleanup in background."""
    def loop():
        while True:
            try:
                _cleanup_old_files()
            except Exception as e:
                log.warning("Cleanup error: %s", e)
            time.sleep(CLEANUP_INTERVAL_HOURS * 3600)

    t = threading.Thread(target=loop, daemon=True)
    t.start()


def _extract_pixel_scale(file_path):
    """Extract pixel scale (nm/px) from TIFF metadata if available.

    Checks ImageJ metadata, OME-TIFF, and standard TIFF resolution tags.
    Returns (scale_nm_per_px, unit_str) or (None, None) if not found.
    """
    path = Path(file_path)
    if path.suffix.lower() not in (".tif", ".tiff"):
        return None, None

    try:
        with tifffile.TiffFile(str(path)) as tif:
            # Check ImageJ metadata
            if tif.imagej_metadata:
                ij = tif.imagej_metadata
                unit = ij.get("unit", "")
                spacing = ij.get("spacing", None)
                if spacing and unit:
                    # Convert to nm
                    if unit in ("nm", "nanometer"):
                        return float(spacing), "nm"
                    elif unit in ("um", "µm", "micron", "micrometer"):
                        return float(spacing) * 1000, "nm"
                    elif unit in ("mm",):
                        return float(spacing) * 1e6, "nm"

            # Check standard TIFF resolution tags
            for page in tif.pages[:1]:
                tags = page.tags
                res_unit_tag = tags.get("ResolutionUnit")
                x_res_tag = tags.get("XResolution")
                if res_unit_tag and x_res_tag:
                    res_unit = res_unit_tag.value  # 1=none, 2=inch, 3=centimeter
                    x_res = x_res_tag.value
                    if isinstance(x_res, tuple):
                        x_res = x_res[0] / x_res[1] if x_res[1] != 0 else 0
                    if x_res > 0 and res_unit == 3:  # centimeter
                        px_per_cm = x_res
                        nm_per_px = 1e7 / px_per_cm
                        if nm_per_px < 1e6:  # sanity check
                            return nm_per_px, "nm"
                    elif x_res > 0 and res_unit == 2:  # inch
                        px_per_inch = x_res
                        nm_per_px = 25.4e6 / px_per_inch
                        if nm_per_px < 1e6:
                            return nm_per_px, "nm"
    except Exception:
        pass

    return None, None


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
    """Check which models are available (at least one checkpoint per structure)."""
    available = {}
    for struct, paths in config.CHECKPOINTS.items():
        available[struct] = any(p.exists() for p in paths)
    return available


def _gpu_cpu_status_html():
    """Generate GPU and CPU status HTML."""
    parts = []

    # GPU status
    try:
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(config.GPU_ID)
            mem_used = torch.cuda.memory_allocated(config.GPU_ID) / 1024**2
            mem_peak = torch.cuda.max_memory_allocated(config.GPU_ID) / 1024**2
            mem_total = torch.cuda.get_device_properties(config.GPU_ID).total_memory / 1024**2
            gpu_pct = (mem_used / mem_total * 100) if mem_total > 0 else 0
            parts.append(
                f'<span style="color:#2ecc71;">GPU: {name}</span> — '
                f'{mem_used:.0f} MB / {mem_total:.0f} MB ({gpu_pct:.0f}%) '
                f'<span style="color:#94a3b8;">[peak: {mem_peak:.0f} MB]</span>'
            )
        else:
            parts.append('<span style="color:#e74c3c;">No GPU available</span>')
    except Exception:
        parts.append('<span style="color:#95a5a6;">GPU status unavailable</span>')

    # CPU status
    try:
        cpu_pct = psutil.cpu_percent(interval=0)
        mem = psutil.virtual_memory()
        cpu_mem_used = mem.used / 1024**3
        cpu_mem_total = mem.total / 1024**3
        parts.append(
            f'CPU: {cpu_pct:.0f}% — '
            f'RAM: {cpu_mem_used:.1f} / {cpu_mem_total:.1f} GB ({mem.percent:.0f}%)'
        )
    except Exception:
        parts.append('<span style="color:#95a5a6;">CPU status unavailable</span>')

    return '<br>'.join(parts)


def process_image(file, structures, mito_thresh, mito_min_area,
                   mem_thresh, mem_min_area, pixel_scale, progress=gr.Progress()):
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

    # Determine pixel scale: manual entry > metadata > none
    scale_nm = None
    if pixel_scale and pixel_scale > 0:
        scale_nm = float(pixel_scale)
    else:
        auto_scale, _ = _extract_pixel_scale(file)
        if auto_scale:
            scale_nm = auto_scale

    progress(0.2, desc=f"Image loaded: {w}x{h}")

    # Build per-structure threshold and min_area dicts
    thresholds = {}
    min_areas = {}
    if "mitochondria" in selected:
        thresholds["mitochondria"] = mito_thresh
        min_areas["mitochondria"] = int(mito_min_area)
    if "membrane" in selected:
        thresholds["membrane"] = mem_thresh
        min_areas["membrane"] = int(mem_min_area)

    # Run segmentation
    progress(0.3, desc="Running segmentation...")
    results = inference.segment(img_gray, structures=selected,
                                thresholds=thresholds, min_areas=min_areas)

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

    # Compute morphometric metrics — side-by-side HTML
    metrics = visualization.compute_morphometrics(results, img_gray.shape, scale_nm=scale_nm)
    elapsed = time.time() - t0
    metrics_html = visualization.format_morphometrics_html(metrics, scale_nm=scale_nm)
    scale_info = f" &nbsp;|&nbsp; Scale: {scale_nm:.1f} nm/px" if scale_nm else ""
    metrics_html += (
        f'<div style="margin-top:12px; color:#94a3b8; font-size:0.85em;">'
        f'Processing time: {elapsed:.1f}s &nbsp;|&nbsp; Image size: {w}x{h}{scale_info}'
        f'</div>'
    )

    # Build download files — descriptive names, TIF format for images
    download_files = []
    zip_contents = []  # paths for zip

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
        metrics_html,
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

    # Per-structure defaults from config
    mito_cfg = config.STRUCTURES["mitochondria"]
    mem_cfg = config.STRUCTURES["membrane"]

    css = """
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400;1,700&display=swap');
    :root {
        --color-accent: #2563eb;
        --color-accent-soft: #dbeafe;
    }
    *, body, button, input, select, textarea, .gradio-container, .gradio-container * {
        font-family: 'Space Mono', monospace !important;
    }
    .model-status { font-size: 0.9em; padding: 8px 12px; background: #f8fafc; border-radius: 6px; }
    .thiru-header { margin-bottom: 8px; }
    .thiru-header svg { max-height: 80px; width: auto; display: block; }
    .thiru-subtitle { color: #64748b; font-size: 0.95em; margin-top: 4px; letter-spacing: 0.5px; padding-left: 4.2%; }
    .thiru-policy { color: #94a3b8; font-size: 0.8em; line-height: 1.5; margin-top: 10px; padding-left: 4.2%; }
    .thiru-policy p { margin: 4px 0; }
    .thiru-policy strong { color: #64748b; }
    footer { display: none !important; }
    /* Hide Gradio PWA install banner */
    .pwa-install-container, .pwa-toast, [class*="pwa"] { display: none !important; }
    /* Settings accordion styling */
    .settings-section { margin-top: 4px; }
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

                # Per-structure settings
                with gr.Accordion("Mitochondria Settings", open=False):
                    mito_thresh = gr.Slider(
                        minimum=0.05, maximum=0.9, value=mito_cfg["threshold"],
                        step=0.05, label="Threshold",
                        info="Lower = more sensitive, higher = more specific",
                    )
                    mito_min_area = gr.Number(
                        value=mito_cfg["min_area"], label="Min Instance Area (px)",
                        info="Instances smaller than this are removed",
                        precision=0,
                    )

                with gr.Accordion("Membrane Settings", open=False):
                    mem_thresh = gr.Slider(
                        minimum=0.05, maximum=0.9, value=mem_cfg["threshold"],
                        step=0.05, label="Threshold",
                        info="Lower = more sensitive, higher = more specific",
                    )
                    mem_min_area = gr.Number(
                        value=mem_cfg["min_area"], label="Min Instance Area (px)",
                        info="Instances smaller than this are removed",
                        precision=0,
                    )

                with gr.Accordion("Scale Settings", open=False):
                    pixel_scale = gr.Number(
                        value=0, label="Pixel Scale (nm/px)",
                        info="Set to 0 for auto-detect from TIFF metadata. If no metadata, pixel units are used.",
                        precision=2,
                    )

                run_btn = gr.Button("Run Segmentation", variant="primary", size="lg")

                # GPU/CPU status with auto-refresh
                gpu_status = gr.HTML(
                    value=f'<div style="margin-top:12px; font-size:0.85em; color:#64748b;">{_gpu_cpu_status_html()}</div>',
                )

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

                # Morphometric analysis (side-by-side HTML)
                results_html = gr.HTML(label="Morphometric Analysis")

                # Downloads
                download_files = gr.File(
                    label="Download Results",
                    file_count="multiple",
                    interactive=False,
                )

        # Wire up the run button
        run_btn.click(
            fn=process_image,
            inputs=[file_input, structures_input,
                    mito_thresh, mito_min_area, mem_thresh, mem_min_area,
                    pixel_scale],
            outputs=[
                input_image,
                mito_overlay,
                membrane_overlay,
                results_html,
                download_files,
            ],
        )

        # GPU/CPU status auto-refresh every 3 seconds
        gpu_timer = gr.Timer(value=3)
        gpu_timer.tick(
            fn=lambda: f'<div style="margin-top:12px; font-size:0.85em; color:#64748b;">{_gpu_cpu_status_html()}</div>',
            outputs=[gpu_status],
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
    function removePWA() {
        // Remove manifest link (prevents "Open in app" / PWA install prompt)
        document.querySelectorAll('link[rel="manifest"]').forEach(el => el.remove());
        // Remove any PWA-related meta tags
        document.querySelectorAll('meta[name="mobile-web-app-capable"], meta[name="apple-mobile-web-app-capable"]').forEach(el => el.remove());
        // Hide any Gradio PWA install banners
        document.querySelectorAll('.pwa-install-container, .pwa-toast, [class*="pwa-"]').forEach(el => el.remove());
    }
    setFavicon();
    removePWA();
    // Watch for Gradio re-injecting its favicon, manifest, or PWA elements
    new MutationObserver(() => {
        const badFav = document.querySelector('link[rel*="icon"]:not([href*="/favicon.svg"])');
        const manifest = document.querySelector('link[rel="manifest"]');
        const pwa = document.querySelector('.pwa-install-container, [class*="pwa-"]');
        if (badFav || manifest || pwa) { setFavicon(); removePWA(); }
    }).observe(document.head, { childList: true });
    new MutationObserver(() => {
        const pwa = document.querySelector('.pwa-install-container, [class*="pwa-"]');
        if (pwa) { removePWA(); }
    }).observe(document.body, { childList: true, subtree: true });
}
"""


def _patch_login_favicon(app):
    """Monkey-patch Gradio's login route to inject our favicon into the login page HTML."""
    favicon_tag = '<link rel="icon" type="image/svg+xml" href="/favicon.svg?v=3">'
    manifest_override = '<link rel="manifest" href="data:application/json,{}">'
    font_link = '<link href="https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400;1,700&display=swap" rel="stylesheet">'
    font_style = "<style>*,body,button,input,select,textarea{font-family:'Space Mono',monospace!important;}</style>"
    inject = favicon_tag + manifest_override + font_link + font_style

    # Find and wrap the login route handler
    for route in app.app.router.routes:
        path = getattr(route, "path", "")
        if path in ("/login", "/login/"):
            original_endpoint = route.endpoint

            async def patched_login(request, _orig=original_endpoint):
                response = await _orig(request)
                if hasattr(response, "body"):
                    body = response.body.decode("utf-8", errors="replace")
                    if "</head>" in body:
                        body = body.replace("</head>", inject + "</head>")
                        from starlette.responses import HTMLResponse
                        return HTMLResponse(content=body, status_code=response.status_code)
                return response

            route.endpoint = patched_login
            break


def main():
    from starlette.responses import JSONResponse, Response
    from starlette.routing import Route

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    app = create_app()

    # Start background cleanup (uploads 24h, results 30d)
    _start_cleanup_thread()

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

    # Patch Gradio's login page to inject our favicon
    _patch_login_favicon(app)

    # Block the main thread
    import threading
    threading.Event().wait()


if __name__ == "__main__":
    main()
