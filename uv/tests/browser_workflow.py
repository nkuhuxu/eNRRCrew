import os
from pathlib import Path

from playwright.sync_api import Locator, expect, sync_playwright


def visible(locator: Locator) -> Locator:
    for index in range(locator.count()):
        candidate = locator.nth(index)
        if candidate.is_visible():
            return candidate
    raise AssertionError(f"No visible element found for locator: {locator}")


output = Path(__file__).resolve().parents[1] / "runtime" / "browser-workflow.png"
errors: list[str] = []
chrome = Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe")

with sync_playwright() as playwright:
    browser = playwright.chromium.launch(
        headless=True, executable_path=str(chrome) if chrome.exists() else None
    )
    page = browser.new_page(viewport={"width": 1440, "height": 1100})
    page.set_default_timeout(90_000)
    page.on(
        "console",
        lambda message: errors.append(message.text) if message.type == "error" else None,
    )
    page.goto(os.getenv("ENRRCREW_E2E_URL", "http://127.0.0.1:8507"))
    page.wait_for_load_state("networkidle")
    tabs = page.get_by_role("tab")

    tabs.nth(1).click()
    page.get_by_text("Yield screening", exact=True).wait_for()
    visible(page.get_by_label("Applied potential (V)")).fill("-0.3")
    visible(page.get_by_label("Electrocatalyst")).fill("Fe-N-C")
    visible(page.get_by_label("Elements (comma-separated symbols)")).fill("Fe, N, C")
    visible(page.get_by_label("Morphology")).fill("porous nanosheet")
    visible(page.get_by_role("button", name="Validate & predict")).click()
    yield_classification = page.get_by_text("Classification", exact=True).first
    yield_error = page.get_by_text("Prediction failed:", exact=False).first
    yield_classification.or_(yield_error).wait_for()
    assert not yield_error.is_visible(), yield_error.text_content()

    tabs.nth(2).click()
    page.get_by_text("FE screening", exact=True).wait_for()
    visible(page.get_by_label("Applied potential (V)")).fill("-0.3")
    visible(page.get_by_label("Electrocatalyst")).fill("Fe-N-C")
    visible(page.get_by_label("Elements (comma-separated symbols)")).fill("Fe, N, C")
    visible(page.get_by_label("Morphology")).fill("porous nanosheet")
    visible(page.get_by_role("button", name="Validate & predict")).click()
    classifications = page.get_by_text("Classification", exact=True)
    fe_error = page.get_by_text("Prediction failed:", exact=False).first
    classifications.nth(1).or_(fe_error).wait_for()
    assert not fe_error.is_visible(), fe_error.text_content()

    tabs.nth(3).click()
    page.get_by_text("Isolated CSV workbench", exact=True).wait_for()
    code = (
        "import pandas as pd\n"
        "df = pd.read_csv('/data/dataset.csv')\n"
        "print(f'ui_rows={len(df)}')\n"
    )
    code_editor = visible(page.get_by_label("Sandbox code", exact=False))
    code_editor.fill(code)
    code_editor.press("Tab")
    run_button = visible(page.get_by_role("button", name="Run in Docker sandbox"))
    expect(run_button).to_be_enabled(timeout=30_000)
    run_button.click()
    page.get_by_text("ui_rows=1117", exact=False).wait_for()
    page.get_by_text("Exit code: 0", exact=False).wait_for()

    page.screenshot(path=str(output), full_page=True)
    browser.close()

assert not errors, f"Browser console errors: {errors}"
print(f"Browser workflow passed; screenshot: {output}")
