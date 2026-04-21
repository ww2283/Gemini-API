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
import os
import platform
import re
import shutil
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import parse_qs

from loguru import logger

from . import jspb_cache
from .capture_path import resolve_capture_path
from .template_capture import capture_all_models

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
    cache_path: Path | None = None,
) -> tuple:
    """Harvest a fresh WAA/BotGuard attestation token via Playwright.

    Args:
        cookies: httpx Cookies jar with Google authentication cookies.
        timeout: Maximum time in seconds to wait for token extraction.
        cache_path: Override path for the per-model jspb templates cache. If
            ``None``, the default cache location is used. If the cache is
            fresh, the per-model capture loop is skipped entirely.

    Returns:
        7-tuple of (token, browser_version, model_ids, botguard_hash,
        reference_inner, reference_headers, per_model_templates) where
        ``per_model_templates`` is either the cached templates dict (on
        cache hit) or the result of ``capture_all_models`` (on cache
        miss/stale). ``None`` if capture produced nothing.

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

    # Consult jspb cache; on fresh hit, skip the per-model capture loop.
    cached_templates: dict[str, str] | None = None
    try:
        cache_entry = jspb_cache.read_cache(path=cache_path)
    except Exception:
        cache_entry = None
    if cache_entry is not None:
        templates = cache_entry.get("templates")
        if isinstance(templates, dict):
            cached_templates = dict(templates)

    token: str | None = None
    botguard_hash: str | None = None
    reference_inner: list | None = None
    reference_headers: dict[str, str] = {}
    token_event = asyncio.Event()

    # Headers the drift detector and auto-patch care about. Captured from
    # the same intercepted StreamGenerate request that yields the WAA token
    # — no extra Chrome roundtrips.
    _TRACKED_REQUEST_HEADERS: tuple[str, ...] = (
        "x-goog-ext-525001261-jspb",
        "x-goog-ext-73010989-jspb",
        "x-goog-ext-73010990-jspb",
    )

    def _extract_from_request(request) -> bool:
        nonlocal token, botguard_hash, reference_inner
        if token_event.is_set():
            return True
        try:
            url = getattr(request, "url", None)
            if isinstance(url, str) and url and "/StreamGenerate" not in url:
                return False
            post_data = request.post_data
            if not post_data:
                return False
            parsed = _parse_stream_generate_request(post_data)
            if parsed is None:
                return False
            candidate = parsed["token"]
            if not (isinstance(candidate, str) and candidate.startswith("!")):
                return False
            token = candidate
            hash_candidate = parsed["botguard_hash"]
            if isinstance(hash_candidate, str):
                botguard_hash = hash_candidate
            reference_inner = parsed["reference_inner"]
            req_headers = request.headers or {}
            for hname in _TRACKED_REQUEST_HEADERS:
                value = req_headers.get(hname) or req_headers.get(hname.lower())
                if isinstance(value, str):
                    reference_headers[hname] = value
            token_event.set()
            return True
        except Exception:
            return False

    async def _handle_route(route):
        try:
            _extract_from_request(route.request)
        except Exception:
            pass
        await route.abort()

    def _handle_request(request):
        _extract_from_request(request)

    # Resolve capture transport BEFORE entering playwright context. The CDP
    # path reuses an already-running user Chrome (signed-in state available
    # for Pro/Thinking); the fresh-launch path keeps today's behaviour.
    cdp_url_override = os.environ.get("GEMINI_WAA_CHROME_URL")
    source, url_or_path = resolve_capture_path(
        cdp_url_override=cdp_url_override,
        managed_profile_path=None,
    )

    browser = None
    owns_browser = False
    try:
        async with async_playwright() as p:
            if source == "cdp":
                logger.debug(
                    f"WAA harvester: connecting over CDP at {url_or_path}"
                )
                browser = await p.chromium.connect_over_cdp(url_or_path)
                if browser.contexts:
                    context = browser.contexts[0]
                else:
                    context = await browser.new_context()
                owns_browser = False
                # Skip cookie injection — the user's Chrome already carries
                # valid sign-in state.
            else:
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
                owns_browser = True

                context = await browser.new_context()
                # Add cookies one-by-one to skip any that Playwright rejects
                for cookie in pw_cookies:
                    try:
                        await context.add_cookies([cookie])
                    except Exception:
                        pass  # Skip invalid cookies silently

            page = await context.new_page()

            # Intercept StreamGenerate to extract the WAA token.
            # ``page.route`` blocks the request but is unreliable on
            # CDP-connected contexts. ``page.on('request')`` is an observer
            # that fires on both fresh-launch and CDP contexts.
            await page.route("**/StreamGenerate*", _handle_route)
            try:
                page.on("request", _handle_request)
            except Exception:
                pass

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

            # Type a trivial message and submit to trigger StreamGenerate.
            logger.debug("WAA harvester: typing prompt to trigger StreamGenerate")
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

            # Per-model jspb template capture: only run on cache miss/stale.
            # On a fresh cache hit, reuse the cached templates to skip the
            # expensive mode-switch loop (~30s for 3 models).
            per_model_templates: dict[str, str] | None = cached_templates
            if cached_templates is None:
                try:
                    captured = await capture_all_models(page)
                except Exception as capture_err:
                    logger.debug(
                        f"per-model jspb capture failed: {capture_err}"
                    )
                    captured = None
                if captured:
                    per_model_templates = captured
                    try:
                        jspb_cache.write_cache(
                            captured, source="fresh", path=cache_path
                        )
                    except Exception as write_err:
                        logger.debug(
                            f"jspb_cache.write_cache failed: {write_err}"
                        )

            return (
                token,
                browser_version,
                model_ids,
                botguard_hash,
                reference_inner,
                reference_headers,
                per_model_templates,
            )

    except WAATokenError:
        raise
    except Exception as e:
        raise WAATokenError(f"Token harvesting failed: {e}") from e
    finally:
        if browser and owns_browser:
            await browser.close()
