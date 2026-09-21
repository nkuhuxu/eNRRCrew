import os
from pathlib import Path

from playwright.sync_api import sync_playwright

output = Path(__file__).resolve().parents[1] / "runtime" / "browser-smoke.png"
errors: list[str] = []
browser_candidates = (
    Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
)
system_browser = next((path for path in browser_candidates if path.exists()), None)

with sync_playwright() as playwright:
    launch_options = {"headless": True}
    if system_browser is not None:
        launch_options["executable_path"] = str(system_browser)
    browser = playwright.chromium.launch(**launch_options)
    page = browser.new_page(viewport={"width": 1440, "height": 1000})
    page.on("console", lambda message: errors.append(message.text) if message.type == "error" else None)
    page.goto(os.getenv("ENRRCREW_E2E_URL", "http://127.0.0.1:8507"))
    page.wait_for_load_state("networkidle")
    page.get_by_text("eNRRCrew", exact=False).first.wait_for()
    tabs = page.get_by_role("tab")
    assert tabs.count() == 5

    tabs.nth(1).click()
    page.get_by_text("Yield screening", exact=True).wait_for()
    tabs.nth(2).click()
    page.get_by_text("FE screening", exact=True).wait_for()
    tabs.nth(3).click()
    page.get_by_text("Isolated CSV workbench", exact=True).wait_for()
    tabs.nth(4).click()
    page.get_by_text("Catalyst recommendation", exact=True).wait_for()
    page.get_by_role("button", name="Run recommendation").wait_for()

    page.screenshot(path=str(output), full_page=True)
    browser.close()

assert not errors, f"Browser console errors: {errors}"
print(f"Browser smoke test passed; screenshot: {output}")
