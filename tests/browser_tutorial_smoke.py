from __future__ import annotations

import os
from pathlib import Path

from playwright.sync_api import Error, sync_playwright

base_url = os.getenv("ENRRCREW_BROWSER_URL", "http://127.0.0.1:8518")
screenshot = Path("runtime") / "ui_tutorial_smoke.png"
screenshot.parent.mkdir(parents=True, exist_ok=True)

with sync_playwright() as playwright:
    try:
        browser = playwright.chromium.launch(headless=True)
    except Error:
        browser = playwright.chromium.launch(channel="chrome", headless=True)
    page = browser.new_page(viewport={"width": 1600, "height": 1000})
    console_errors: list[str] = []
    page.on(
        "console",
        lambda message: console_errors.append(message.text)
        if message.type == "error"
        else None,
    )
    page.goto(base_url, wait_until="networkidle", timeout=60_000)

    page.get_by_text("How to use", exact=True).click()
    page.get_by_text("Choose a workspace based on your goal", exact=True).wait_for()
    page.get_by_role("tab", name="⌘ Knowledge base update").click()
    page.get_by_text("Upload and validate a literature table", exact=False).wait_for()
    page.get_by_text("Activate selected version", exact=False).wait_for()
    page.screenshot(path=str(screenshot), full_page=True)

    assert not console_errors, console_errors
    browser.close()

print(f"Tutorial UI smoke test passed; screenshot: {screenshot}")
