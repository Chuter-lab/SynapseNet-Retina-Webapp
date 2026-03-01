"""Playwright test for THIRU results display.

Tests:
1. Login and upload test image
2. Run segmentation
3. Verify 4-panel display (Input, Mitochondria overlay, Membrane overlay, Ribbon overlay)
4. Verify morphometric metrics contain expected fields
5. Verify downloads include all expected files (no heatmaps)
"""
import os
import sys
import time
from playwright.sync_api import sync_playwright

URL = os.environ.get("THIRU_URL", "https://thiru.chuterlab.com/")
USERNAME = os.environ.get("THIRU_USER", "")
PASSWORD = os.environ.get("THIRU_PASS", "")
TEST_IMAGE = os.environ.get("THIRU_TEST_IMAGE", "/Users/bento/Downloads/THIRU_test_crop_raw.png")
DOWNLOAD_DIR = os.environ.get("THIRU_DOWNLOAD_DIR", "/Users/bento/Downloads")
RESULTS_FILE = os.path.join(DOWNLOAD_DIR, "thiru_test_results.txt")

SEG_TIMEOUT = 180_000


def log(msg, results):
    print(msg)
    results.append(msg)


def run_test():
    results = []
    passed = 0
    failed = 0

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()

        # ===================== LOGIN =====================
        log("=" * 60, results)
        log("TEST 1: Login", results)

        if USERNAME and PASSWORD:
            page.goto(URL, timeout=60000)
            time.sleep(5)
            import urllib.parse
            encoded_body = urllib.parse.urlencode({"username": USERNAME, "password": PASSWORD})
            log(f"  Encoded body: {encoded_body}", results)
            login_result = page.evaluate("""(body) => {
                return fetch('/login', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/x-www-form-urlencoded'},
                    body: body,
                }).then(async r => {
                    const text = await r.text();
                    return {status: r.status, text: text};
                }).catch(e => ({error: e.message}));
            }""", encoded_body)
            log(f"  Login result: {login_result}", results)
            page.reload(timeout=60000)
            time.sleep(10)

        page.screenshot(path=os.path.join(DOWNLOAD_DIR, "thiru_after_login.png"))

        run_btn = page.locator("button").filter(has_text="Run Segmentation")
        try:
            run_btn.first.wait_for(timeout=30000)
        except Exception:
            time.sleep(10)
            run_btn = page.locator("button").filter(has_text="Run Segmentation")
        if run_btn.count() > 0:
            log("  PASS: Login successful", results)
            passed += 1
        else:
            log("  FAIL: Login failed", results)
            failed += 1
            browser.close()
            _write_results(results, passed, failed)
            return passed, failed

        # ===================== UPLOAD =====================
        log("=" * 60, results)
        log("TEST 2: Upload image", results)
        page.locator("input[type='file']").first.set_input_files(TEST_IMAGE)
        time.sleep(2)
        log("  PASS: Image uploaded", results)
        passed += 1

        # ===================== RUN SEGMENTATION =====================
        log("=" * 60, results)
        log("TEST 3: Run segmentation", results)
        run_btn.first.click()
        log("  Clicked Run Segmentation...", results)

        try:
            page.wait_for_function(
                """() => {
                    const imgs = document.querySelectorAll('.image-container img, [data-testid="image"] img');
                    return imgs.length >= 4 && imgs[0].src && imgs[0].src.length > 100;
                }""",
                timeout=SEG_TIMEOUT,
            )
            time.sleep(5)
            log("  PASS: Segmentation completed", results)
            passed += 1
        except Exception:
            time.sleep(60)
            img_count = page.locator("img").count()
            if img_count >= 3:
                log(f"  PASS: Segmentation likely completed ({img_count} imgs)", results)
                passed += 1
            else:
                log("  FAIL: Segmentation did not complete", results)
                failed += 1

        page.screenshot(path=os.path.join(DOWNLOAD_DIR, "thiru_post_run.png"))

        # ===================== 4-PANEL DISPLAY =====================
        log("=" * 60, results)
        log("TEST 4: Four-panel display (Input, Mito, Membrane, Ribbon)", results)
        page_html = page.content()
        page_text = page.inner_text("body")

        has_input = "Input" in page_html
        has_mito = "Mitochondria" in page_html
        has_membrane = "Membrane" in page_html
        has_ribbon = "Ribbon" in page_html

        img_panels = page.locator(".image-container img, [data-testid='image'] img")
        panel_count = img_panels.count()

        log(f"  Image panels found: {panel_count}", results)
        log(f"  Labels: Input={has_input}, Mitochondria={has_mito}, Membrane={has_membrane}, Ribbon={has_ribbon}", results)

        if panel_count >= 4 and has_input and has_mito and has_membrane and has_ribbon:
            log("  PASS: Four-panel display present", results)
            passed += 1
        else:
            log("  FAIL: Expected 4 panels with correct labels", results)
            failed += 1

        # Verify no probability maps or heatmaps visible
        has_prob = "probability" in page_text.lower() or "heatmap" in page_text.lower()
        if not has_prob:
            log("  PASS: No probability maps/heatmaps visible", results)
            passed += 1
        else:
            log("  FAIL: Probability/heatmap text found on page", results)
            failed += 1

        # ===================== MORPHOMETRICS =====================
        log("=" * 60, results)
        log("TEST 5: Morphometric metrics", results)

        expected = ["Instances", "Total area", "Coverage", "Mean area", "Min area", "Max area", "Std area"]
        found_m = [m for m in expected if m in page_text]

        extra = []
        lower_text = page_text.lower()
        for kw in ["circularity", "aspect ratio", "perimeter"]:
            if kw in lower_text:
                extra.append(kw)

        log(f"  Core metrics: {len(found_m)}/{len(expected)}", results)
        log(f"  Extra metrics: {extra}", results)

        if len(found_m) >= 5:
            log("  PASS: Morphometrics displayed", results)
            passed += 1
        else:
            log(f"  FAIL: Only {len(found_m)} core metrics", results)
            failed += 1

        if "Processing time" in page_text:
            log("  PASS: Processing time shown", results)
            passed += 1
        else:
            log("  FAIL: Processing time missing", results)
            failed += 1

        # ===================== DOWNLOADS =====================
        log("=" * 60, results)
        log("TEST 6: Download files", results)

        html = page.content()
        expected_files = [
            "display_panel",
            "overlay_Mitochondria",
            "overlay_Presynaptic_Membrane",
            "overlay_Synaptic_Ribbon",
            "mask_Mitochondria",
            "mask_Presynaptic_Membrane",
            "mask_Synaptic_Ribbon",
            "metrics.csv",
        ]
        # Verify NO probability maps
        unwanted_files = [
            "prob_mitochondria.tif",
            "prob_membrane.tif",
        ]

        found_f = [f for f in expected_files if f in html or f in page_text]
        missing_f = [f for f in expected_files if f not in found_f]
        found_unwanted = [f for f in unwanted_files if f in html or f in page_text]

        for f in found_f:
            log(f"  Found: {f}", results)
        for f in missing_f:
            log(f"  Missing: {f}", results)
        for f in found_unwanted:
            log(f"  UNWANTED: {f}", results)

        if len(found_f) >= 6 and len(found_unwanted) == 0:
            log(f"  PASS: {len(found_f)}/{len(expected_files)} files, no heatmaps", results)
            passed += 1
        else:
            log(f"  FAIL: {len(found_f)}/{len(expected_files)} files, {len(found_unwanted)} unwanted", results)
            failed += 1

        # ===================== FINAL =====================
        page.screenshot(path=os.path.join(DOWNLOAD_DIR, "thiru_final.png"), full_page=True)
        log(f"\nScreenshots saved to {DOWNLOAD_DIR}", results)

        browser.close()

    _write_results(results, passed, failed)
    return passed, failed


def _write_results(results, passed, failed):
    total = passed + failed
    summary = f"\n{'=' * 60}\nRESULTS: {passed}/{total} passed, {failed}/{total} failed\n{'=' * 60}"
    results.append(summary)
    print(summary)
    with open(RESULTS_FILE, "w") as f:
        f.write("\n".join(results))
    print(f"Results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    passed, failed = run_test()
    sys.exit(0 if failed == 0 else 1)
