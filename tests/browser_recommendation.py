import os
from pathlib import Path

from playwright.sync_api import Locator, sync_playwright


def visible(locator: Locator) -> Locator:
    for index in range(locator.count()):
        candidate = locator.nth(index)
        if candidate.is_visible():
            return candidate
    raise AssertionError(f"No visible element found for locator: {locator}")


output = Path(__file__).resolve().parents[1] / "runtime" / "browser-recommendation.png"
errors: list[str] = []
chrome = Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe")

with sync_playwright() as playwright:
    browser = playwright.chromium.launch(
        headless=True, executable_path=str(chrome) if chrome.exists() else None
    )
    page = browser.new_page(viewport={"width": 1440, "height": 1100})
    page.set_default_timeout(180_000)
    page.on(
        "console",
        lambda message: errors.append(message.text) if message.type == "error" else None,
    )
    page.goto(os.getenv("ENRRCREW_E2E_URL", "http://127.0.0.1:8507"))
    page.wait_for_load_state("networkidle")
    page.get_by_role("tab").nth(4).click()
    page.get_by_text("Catalyst recommendation", exact=True).wait_for()
    visible(page.get_by_role("button", name="Run recommendation")).click()
    page.get_by_text("Strict double-high", exact=True).wait_for(state="attached")
    page.get_by_role("tab").nth(4).click()
    page.get_by_text("Strict double-high", exact=True).wait_for()
    page.get_by_text("Strict recommendations", exact=True).wait_for()
    page.get_by_text("Near misses", exact=True).wait_for()
    page.screenshot(path=str(output), full_page=True)
    browser.close()

assert not errors, f"Browser console errors: {errors}"
print(f"Recommendation browser workflow passed; screenshot: {output}")
