"""
WAA/BotGuard token harvester using Playwright.

Launches headless Chrome, loads the Gemini page to trigger BotGuard attestation,
types a trivial prompt to trigger StreamGenerate, intercepts the request to extract
the attestation token from inner_req_list[3], then aborts the request.

The token is ~1.3KB, starts with '!', is single-use, and has a 3-5 minute TTL.
It is validated only at stream connection establishment.
"""

from __future__ import annotations

import asyncio
import json
import platform
import shutil
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import parse_qs

from loguru import logger

if TYPE_CHECKING:
    from httpx import Cookies


def _cookies_for_playwright(httpx_cookies) -> list[dict]:
    """Convert httpx.Cookies to Playwright's list-of-dicts format."""
    result = []
    for cookie in httpx_cookies.jar:
        entry = {
            "name": cookie.name,
            "value": cookie.value,
            "domain": cookie.domain,
            "path": cookie.path,
        }
        # Playwright requires 'url' or 'domain' — domain from the jar suffices.
        # But domain must start with '.' for cross-subdomain cookies.
        if entry["domain"] and not entry["domain"].startswith("."):
            entry["domain"] = "." + entry["domain"]
        result.append(entry)
    return result


def _find_system_chrome() -> str | None:
    """Find system Chrome installation path."""
    system = platform.system()
    if system == "Darwin":
        p = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
        if p.exists():
            return str(p)
    elif system == "Linux":
        for name in ("google-chrome", "google-chrome-stable"):
            path = shutil.which(name)
            if path:
                return path
    elif system == "Windows":
        for p in (
            Path("C:/Program Files/Google/Chrome/Application/chrome.exe"),
            Path("C:/Program Files (x86)/Google/Chrome/Application/chrome.exe"),
        ):
            if p.exists():
                return str(p)
    return None


async def harvest_waa_token(cookies: Cookies, timeout: float = 30.0) -> str:
    """Harvest a fresh WAA/BotGuard attestation token via Playwright.

    Args:
        cookies: httpx Cookies jar with Google authentication cookies.
        timeout: Maximum time in seconds to wait for token extraction.

    Returns:
        The WAA token string (starts with '!', ~1.3KB).

    Raises:
        WAATokenError: If token harvesting fails for any reason.
    """
    from ..exceptions import WAATokenError

    try:
        from playwright.async_api import async_playwright
    except ImportError:
        raise WAATokenError(
            "playwright is required for WAA token harvesting. "
            "Install with: pip install playwright"
        )

    pw_cookies = _cookies_for_playwright(cookies)
    if not pw_cookies:
        raise WAATokenError("No cookies available for WAA token harvesting")

    token: str | None = None
    token_event = asyncio.Event()

    async def _handle_route(route):
        nonlocal token
        try:
            request = route.request
            post_data = request.post_data
            if post_data:
                params = parse_qs(post_data)
                f_req = params.get("f.req", [None])[0]
                if f_req:
                    outer = json.loads(f_req)
                    inner = json.loads(outer[1])
                    candidate = inner[3]
                    if isinstance(candidate, str) and candidate.startswith("!"):
                        token = candidate
                        token_event.set()
        except Exception:
            pass
        await route.abort()

    browser = None
    try:
        async with async_playwright() as p:
            # Try channel="chrome" first (uses system Chrome, no download)
            launch_kwargs = dict(
                headless=True,
                args=["--no-sandbox", "--disable-dev-shm-usage"],
            )
            try:
                browser = await p.chromium.launch(channel="chrome", **launch_kwargs)
            except Exception:
                # Fallback: explicit path
                chrome_path = _find_system_chrome()
                if chrome_path:
                    logger.debug(f"Falling back to system Chrome at {chrome_path}")
                    browser = await p.chromium.launch(
                        executable_path=chrome_path, **launch_kwargs
                    )
                else:
                    raise WAATokenError(
                        "System Chrome not found. Install Google Chrome or run "
                        "'playwright install chromium' to download a bundled browser."
                    )

            context = await browser.new_context()
            await context.add_cookies(pw_cookies)

            page = await context.new_page()

            # Intercept StreamGenerate to extract the WAA token
            await page.route("**/StreamGenerate*", _handle_route)

            # Navigate to Gemini
            await page.goto(
                "https://gemini.google.com/app",
                wait_until="networkidle",
                timeout=timeout * 1000,
            )

            # Wait for input area
            input_sel = 'div[contenteditable="true"], textarea'
            await page.wait_for_selector(input_sel, timeout=15000)

            # Type a trivial message and submit to trigger StreamGenerate
            await page.click(input_sel)
            await page.type(input_sel, "hi")
            await page.keyboard.press("Enter")

            # Wait for the token to be captured
            try:
                await asyncio.wait_for(token_event.wait(), timeout=15.0)
            except asyncio.TimeoutError:
                pass

            if not token:
                raise WAATokenError("StreamGenerate request not intercepted or token missing")

            logger.debug(f"WAA token harvested ({len(token)} chars)")
            return token

    except WAATokenError:
        raise
    except Exception as e:
        raise WAATokenError(f"Token harvesting failed: {e}") from e
    finally:
        if browser:
            await browser.close()
