import os
from contextlib import suppress
from getpass import getpass
from pathlib import Path

from playwright.sync_api import Locator, expect, sync_playwright
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError


def visible(locator: Locator) -> Locator:
    for index in range(locator.count()):
        candidate = locator.nth(index)
        if candidate.is_visible():
            return candidate
    raise AssertionError(f"No visible element found for locator: {locator}")


api_key = getpass("Session API key: ").strip()
if not api_key:
    raise SystemExit("An API key is required")

url = os.getenv("ENRRCREW_E2E_URL", "http://127.0.0.1:8508")
output = Path(__file__).resolve().parents[1] / "runtime" / "online-browser-smoke.png"
chrome = Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe")

with sync_playwright() as playwright:
    browser = playwright.chromium.launch(
        headless=True, executable_path=str(chrome) if chrome.exists() else None
    )
    page = browser.new_page(viewport={"width": 1440, "height": 1100})
    page.set_default_timeout(240_000)
    page.goto(url)
    page.wait_for_load_state("networkidle")

    page.get_by_label("API key").fill(api_key)
    page.get_by_label("Base URL").fill("https://api.chatanywhere.tech/v1")
    page.get_by_label("Base URL").press("Tab")
    page.get_by_text("LLM link: ONLINE", exact=False).wait_for()
    page.wait_for_timeout(2_000)

    tabs = page.get_by_role("tab")
    tabs.nth(1).click()
    visible(
        page.get_by_text(
            "Extract input from a natural-language description", exact=True
        )
    ).click()
    description = visible(page.get_by_label("Experimental description"))
    description.fill(
        "Fe-N-C porous nanosheets containing Fe, N and C at -0.45 V in neutral PBS."
    )
    description.press("Tab")
    page.wait_for_timeout(1_000)
    visible(page.get_by_role("button", name="Extract & review")).click()
    extraction_success = page.get_by_text(
        "Input extracted. Review every field", exact=False
    ).first
    extraction_error = page.get_by_text("Extraction failed:", exact=False).first
    with suppress(PlaywrightTimeoutError):
        extraction_success.or_(extraction_error).wait_for(timeout=60_000)
    # Streamlit success notices are transient across widget-state reruns;
    # the populated review form is the durable success condition.
    assert not extraction_error.is_visible(), extraction_error.text_content()
    catalyst = visible(page.get_by_label("Electrocatalyst"))
    expect(catalyst).not_to_have_value("", timeout=30_000)

    tabs.nth(3).click()
    visible(page.get_by_label("Analysis request")).fill(
        "Print the number of rows in the dataset."
    )
    visible(page.get_by_label("Analysis request")).press("Tab")
    page.wait_for_timeout(1_000)
    tabs.nth(3).click()
    page.get_by_text("Isolated CSV workbench", exact=True).wait_for()
    visible(page.get_by_role("button", name="Generate analysis code")).click()
    editor = visible(page.get_by_label("Sandbox code", exact=False))
    editor.wait_for()
    page.wait_for_function(
        "element => element.value.trim().length > 0", arg=editor.element_handle()
    )

    tabs.nth(0).click()
    chat = page.locator('[data-testid="stChatInput"] textarea')
    if not chat.count():
        chat = page.locator('[data-testid="stChatInput"] input')
    visible(chat).fill("Name one catalyst discussed in the eNRR knowledge graph.")
    visible(chat).press("Enter")
    page.get_by_text("Retrieval failed:", exact=False).wait_for(state="hidden")
    page.locator('[data-testid="stChatMessage"]').nth(1).wait_for()
    assistant_text = page.locator('[data-testid="stChatMessage"]').nth(1).inner_text()
    assert assistant_text.strip()
    assert "Retrieval failed:" not in assistant_text
    visible(page.get_by_text("Agent trace", exact=True)).wait_for()

    page.screenshot(path=str(output), full_page=True)
    browser.close()

print(f"Online browser workflow passed; screenshot: {output}")
