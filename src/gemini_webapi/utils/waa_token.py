"""
WAA/BotGuard token harvester using Playwright.

Launches headless Chrome, loads the Gemini page to trigger BotGuard attestation,
types a trivial prompt to trigger StreamGenerate, intercepts the request to extract
the attestation token from inner_req_list[3], then aborts the request.

Also extracts model ID arrays from the page's embedded experiment data, enabling
dynamic discovery of current model IDs (which Google rotates periodically).

The token is ~1.3KB, starts with '!', is single-use, and has a 3-5 minute TTL.
It is validated only at stream connection establishment.
"""

from __future__ import annotations

import asyncio
import json
import platform
import re
import shutil
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import parse_qs

from loguru import logger

# RPC names embedded in Gemini's server-side experiment data that contain
# model ID arrays.  These map each "mode" (Fast / Thinking / Pro) to an
# ordered list of model IDs — one per tier/variant.
_MODEL_DISCOVERY_RPCS: dict[str, str] = {
    "pro": "g9Ghwf",
    "flash": "xjRbsb",
    "thinking": "G1FaEb",
}


def _extract_model_ids_from_html(html: str) -> dict[str, list[str]]:
    """Extract model ID arrays from Gemini page HTML.

    The page embeds experiment/feature-flag data containing arrays of hex
    model IDs keyed by RPC name.  Format in HTML (after escaping)::

        "rpc_name",["\\"[[\\\\\\\"id1\\\\\\\",\\\\\\\"id2\\\\\\\"]]\\""]]

    Returns a dict like ``{"pro": ["id1", "id2", ...], ...}``.
    """
    result: dict[str, list[str]] = {}
    for model_type, rpc_name in _MODEL_DISCOVERY_RPCS.items():
        # Find the RPC entry and its payload containing the model ID array
        # Pattern: "rpc_name",[" ... [[\"id1\",\"id2\"]] ... "]
        pattern = re.escape(f'\\"{rpc_name}\\"') + r',\[.+?\[\[(.+?)\]\]'
        match = re.search(pattern, html)
        if not match:
            continue
        # Extract hex model IDs from the matched group
        ids = re.findall(r'[0-9a-f]{16}', match.group(1))
        if ids:
            result[model_type] = ids
            logger.debug(f"Discovered {model_type} model IDs: {ids}")
    return result

if TYPE_CHECKING:
    from curl_cffi.requests import Cookies


def _parse_stream_generate_request(post_data: str) -> dict | None:
    """Parse URL-form-encoded post_data from a StreamGenerate request.

    Returns dict with keys 'token', 'botguard_hash', 'reference_inner'
    on success, or None if post_data cannot be parsed.
    """
    try:
        params = parse_qs(post_data)
        f_req = params.get("f.req", [None])[0]
        if not f_req:
            return None
        outer = json.loads(f_req)
        inner = json.loads(outer[1])
        token = inner[3] if len(inner) > 3 else None
        botguard_hash = inner[4] if len(inner) > 4 else None
        return {
            "token": token,
            "botguard_hash": botguard_hash,
            "reference_inner": inner,
        }
    except Exception:
        return None


def _cookies_for_playwright(httpx_cookies) -> list[dict]:
    """Convert curl_cffi.requests.Cookies to Playwright's list-of-dicts format."""
    result = []
    for cookie in httpx_cookies.jar:
        name = cookie.name
        value = cookie.value or ""
        domain = cookie.domain or ""
        path = cookie.path or "/"

        # Skip cookies without name or domain — Playwright rejects them
        if not name or not domain:
            continue

        # Playwright requires domain to start with '.' for cross-subdomain cookies
        if not domain.startswith("."):
            domain = "." + domain

        result.append({
            "name": name,
            "value": value,
            "domain": domain,
            "path": path,
        })
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


async def harvest_waa_token(
    cookies: Cookies,
    timeout: float = 45.0,
    target_model: str | None = None,
) -> str:
    """Harvest a fresh WAA/BotGuard attestation token via Playwright.

    Args:
        cookies: httpx Cookies jar with Google authentication cookies.
        timeout: Maximum time in seconds to wait for token extraction.

    Returns:
        Tuple of (token, browser_version, model_ids, botguard_hash, reference_inner)
        where token starts with '!' (~1.3KB), browser_version is e.g.
        '146.0.7680.178', and reference_inner is the full decoded inner_req_list
        captured from the intercepted StreamGenerate request.

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
    botguard_hash: str | None = None
    reference_inner: list | None = None
    token_event = asyncio.Event()

    async def _handle_route(route):
        nonlocal token, botguard_hash, reference_inner
        try:
            request = route.request
            post_data = request.post_data
            if post_data:
                parsed = _parse_stream_generate_request(post_data)
                if parsed is not None:
                    candidate = parsed["token"]
                    if isinstance(candidate, str) and candidate.startswith("!"):
                        token = candidate
                        hash_candidate = parsed["botguard_hash"]
                        if isinstance(hash_candidate, str):
                            botguard_hash = hash_candidate
                        reference_inner = parsed["reference_inner"]
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
                logger.debug("WAA harvester: launching Chrome (channel=chrome)")
                browser = await p.chromium.launch(channel="chrome", **launch_kwargs)
            except Exception as launch_err:
                logger.debug(f"WAA harvester: channel=chrome failed: {launch_err}")
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
            # Add cookies one-by-one to skip any that Playwright rejects
            for cookie in pw_cookies:
                try:
                    await context.add_cookies([cookie])
                except Exception:
                    pass  # Skip invalid cookies silently

            page = await context.new_page()

            # Intercept StreamGenerate to extract the WAA token
            await page.route("**/StreamGenerate*", _handle_route)

            # Navigate to Gemini — use domcontentloaded because networkidle
            # never fires (Gemini keeps persistent WebSocket/polling connections)
            logger.debug("WAA harvester: navigating to gemini.google.com/app")
            await page.goto(
                "https://gemini.google.com/app",
                wait_until="domcontentloaded",
                timeout=timeout * 1000,
            )

            # Wait for input area
            input_sel = 'div[contenteditable="true"], textarea'
            logger.debug("WAA harvester: waiting for input selector")
            await page.wait_for_selector(input_sel, timeout=20000)

            # Extract model IDs from page HTML before triggering the request.
            # The page embeds experiment data with model ID arrays per mode.
            model_ids: dict[str, list[str]] = {}
            try:
                html = await page.content()
                model_ids = _extract_model_ids_from_html(html)
            except Exception as e:
                logger.debug(f"WAA harvester: model ID extraction failed: {e}")

            # If caller requested a specific model, drive the mode picker
            # BEFORE typing the warmup prompt so the StreamGenerate we
            # intercept is the one for the target mode.
            if target_model in ("pro", "flash", "thinking"):
                try:
                    await page.click(
                        'button[aria-label="Open mode picker"]', timeout=5000
                    )
                    await page.wait_for_selector(
                        '[role="menuitem"]', timeout=3000
                    )
                    label_for_click = {
                        "pro": "Pro",
                        "flash": "Fast",
                        "thinking": "Thinking",
                    }[target_model]
                    clicked = await page.evaluate(
                        """(label) => {
                            const items = document.querySelectorAll('[role="menuitem"]');
                            for (const el of items) {
                                if (el.textContent.includes(label)) {
                                    el.click();
                                    return true;
                                }
                            }
                            return false;
                        }""",
                        label_for_click,
                    )
                    if not clicked:
                        logger.debug(
                            f"WAA harvester: could not find menu item for target_model={target_model!r}"
                        )
                    else:
                        # Wait for the mode-picker button label to reflect the
                        # new selection. Note: even after the label updates,
                        # the Gemini client's XHR-dispatch state may lag for
                        # short prompts (Google appears to auto-route trivial
                        # messages to a default mode), so the captured
                        # reference may still resemble Flash for Pro/Thinking
                        # targets. Drifts on slots shared across modes are
                        # still caught; Pro-specific slots need DevTools
                        # manual capture (see python -m gemini_webapi.diag).
                        try:
                            await page.wait_for_function(
                                """(label) => {
                                    const btn = document.querySelector(
                                        'button[aria-label="Open mode picker"]'
                                    );
                                    return btn && btn.textContent.trim() === label;
                                }""",
                                arg=label_for_click,
                                timeout=3000,
                            )
                        except Exception:
                            logger.debug(
                                f"WAA harvester: mode-picker label did not commit "
                                f"to {label_for_click!r} within 3s"
                            )
                except Exception as e:
                    logger.debug(
                        f"WAA harvester: mode-picker click failed for target_model={target_model!r}: {e}"
                    )

            # Type a trivial message and submit to trigger StreamGenerate.
            # Use the Send button (not Enter) so the request picks up the
            # currently-selected mode — Enter on a contenteditable div can
            # route through a different code path that ignores the picker.
            logger.debug("WAA harvester: typing prompt to trigger StreamGenerate")
            await page.click(input_sel)
            await page.type(input_sel, "hi")
            try:
                await page.wait_for_selector(
                    'button[aria-label*="Send"]', timeout=3000
                )
                await page.click('button[aria-label*="Send"]', timeout=3000)
            except Exception:
                # Fall back to Enter if the Send button is not reachable.
                await page.keyboard.press("Enter")

            # Wait for the token to be captured
            try:
                await asyncio.wait_for(token_event.wait(), timeout=15.0)
            except asyncio.TimeoutError:
                pass

            if not token:
                raise WAATokenError("StreamGenerate request not intercepted or token missing")

            browser_version = browser.version if browser else None
            logger.debug(
                f"WAA token harvested ({len(token)} chars, "
                f"hash={'yes' if botguard_hash else 'no'}, Chrome {browser_version})"
            )

            # Clean up routes before closing to avoid TargetClosedError noise
            # from in-flight requests that haven't been handled yet.
            try:
                await page.unroute_all(behavior="ignoreErrors")
            except Exception:
                pass

            return token, browser_version, model_ids, botguard_hash, reference_inner

    except WAATokenError:
        raise
    except Exception as e:
        raise WAATokenError(f"Token harvesting failed: {e}") from e
    finally:
        if browser:
            await browser.close()
