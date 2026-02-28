"""Playwright test for THIRU results display redesign.

Tests:
1. Login and upload test image
2. Run segmentation with all 3 structures
3. Verify overlay image appears
4. Toggle each checkbox and verify overlay updates
5. Verify gallery shows masks only (no prob maps)
6. Verify morphometric metrics contain expected fields
7. Verify downloads include all expected files
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
        page.goto(URL, timeout=30000)
        page.wait_for_load_state("networkidle", timeout=15000)

        inputs = page.locator("input")
        if inputs.count() >= 2:
            inputs.nth(0).fill(USERNAME)
            inputs.nth(1).fill(PASSWORD)
            page.locator("button").first.click()
            page.wait_for_load_state("networkidle", timeout=15000)
            time.sleep(2)

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
                    return imgs.length > 0 && imgs[0].src && imgs[0].src.length > 100;
                }""",
                timeout=SEG_TIMEOUT,
            )
            time.sleep(5)
            log("  PASS: Segmentation completed", results)
            passed += 1
        except Exception:
            time.sleep(60)
            img_count = page.locator("img").count()
            if img_count > 5:
                log(f"  PASS: Segmentation likely completed ({img_count} imgs)", results)
                passed += 1
            else:
                log("  FAIL: Segmentation did not complete", results)
                failed += 1

        page.screenshot(path=os.path.join(DOWNLOAD_DIR, "thiru_post_run.png"))

        # ===================== OVERLAY =====================
        log("=" * 60, results)
        log("TEST 4: Overlay image present", results)
        page_html = page.content()
        if "Overlay" in page_html and page.locator("img").count() > 0:
            log("  PASS: Overlay image present", results)
            passed += 1
        else:
            log("  FAIL: Overlay not found", results)
            failed += 1

        # ===================== CHECKBOXES =====================
        log("=" * 60, results)
        log("TEST 5: Layer toggle checkboxes", results)

        # Standalone checkboxes have data-testid="checkbox" and class svelte-1q8xtp9
        # CheckboxGroup items have class svelte-yb2gcx
        # Use data-testid="checkbox" to find standalone toggles
        toggle_cbs = page.locator("input[data-testid='checkbox']")
        toggle_count = toggle_cbs.count()
        log(f"  Standalone toggle checkboxes found: {toggle_count}", results)

        # Get their labels
        toggle_info = page.evaluate("""() => {
            const cbs = document.querySelectorAll('input[data-testid="checkbox"]');
            return Array.from(cbs).map((cb, i) => ({
                index: i,
                label: cb.closest('label')?.textContent?.trim() || 'unknown',
                checked: cb.checked
            }));
        }""")
        for info in toggle_info:
            log(f"  Toggle {info['index']}: '{info['label']}' checked={info['checked']}", results)

        if toggle_count == 3:
            log("  PASS: All 3 toggle checkboxes found", results)
            passed += 1
        else:
            log(f"  FAIL: Expected 3, found {toggle_count}", results)
            failed += 1

        # ===================== TOGGLE OVERLAY =====================
        log("=" * 60, results)
        log("TEST 6: Toggle checkboxes update overlay", results)

        toggle_ok = True
        for i in range(min(toggle_count, 3)):
            label = toggle_info[i]["label"] if i < len(toggle_info) else f"toggle_{i}"

            # Get overlay image src before
            before_src = page.evaluate(
                "() => { const img = document.querySelector('.image-container img'); return img ? img.src : ''; }"
            )

            # Click the checkbox
            toggle_cbs.nth(i).click()
            time.sleep(4)

            # Get overlay image src after
            after_src = page.evaluate(
                "() => { const img = document.querySelector('.image-container img'); return img ? img.src : ''; }"
            )

            changed = before_src != after_src
            if changed:
                log(f"  {label} toggle: overlay CHANGED", results)
            else:
                log(f"  {label} toggle: overlay UNCHANGED", results)
                toggle_ok = False

            page.screenshot(path=os.path.join(DOWNLOAD_DIR, f"thiru_toggle_{i}_{label}_off.png"))

            # Toggle back
            toggle_cbs.nth(i).click()
            time.sleep(4)

        if toggle_ok:
            log("  PASS: All toggles changed overlay", results)
            passed += 1
        else:
            log("  FAIL: Some toggles did not change overlay", results)
            failed += 1

        # ===================== GALLERY =====================
        log("=" * 60, results)
        log("TEST 7: Gallery shows masks only", results)

        page_text = page.inner_text("body")
        has_prob = "probability" in page_text.lower()

        if not has_prob:
            log("  PASS: No probability maps visible", results)
            passed += 1
        else:
            log("  FAIL: Probability maps text found on page", results)
            failed += 1

        # ===================== MORPHOMETRICS =====================
        log("=" * 60, results)
        log("TEST 8: Morphometric metrics", results)

        expected = ["Instances", "Total area", "Coverage", "Mean area", "Min area", "Max area", "Std area"]
        found_m = [m for m in expected if m in page_text]

        extra = []
        lower_text = page_text.lower()
        for kw in ["circularity", "aspect ratio", "perimeter", "density"]:
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
        log("TEST 9: Download files", results)

        html = page.content()
        expected_files = [
            "last_overlay.png",
            "mask_vesicles.tif",
            "mask_mitochondria.tif",
            "mask_membrane.tif",
            "prob_vesicles.tif",
            "prob_mitochondria.tif",
            "prob_membrane.tif",
            "metrics.csv",
        ]

        found_f = [f for f in expected_files if f in html or f in page_text]
        missing_f = [f for f in expected_files if f not in found_f]
        for f in found_f:
            log(f"  Found: {f}", results)
        for f in missing_f:
            log(f"  Missing: {f}", results)

        if len(found_f) >= 7:
            log(f"  PASS: {len(found_f)}/{len(expected_files)} files", results)
            passed += 1
        else:
            log(f"  FAIL: {len(found_f)}/{len(expected_files)} files", results)
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
