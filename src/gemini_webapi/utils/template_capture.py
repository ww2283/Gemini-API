"""Per-model jspb template capture loop for the WAA harvester.

Drives a connected Playwright page through Flash -> Pro -> Thinking mode
switches, submits a trivial prompt per mode, and intercepts the outgoing
StreamGenerate request to lift the ``x-goog-ext-525001261-jspb`` header.
"""
from __future__ import annotations

import asyncio
import random
from typing import Any, Awaitable, Callable, Sequence


CAPTURE_MODELS: tuple[tuple[str, str, str, int], ...] = (
    ("flash", "flash", "Fast", 1),
    ("pro", "pro", "Pro", 3),
    ("thinking", "thinking", "Thinking", 1),
)
# (canonical_key, data_test_id, mode_picker_label, expected_variant_int)

_CAPTURE_TIMEOUT_SECONDS = 15.0

_MODE_PICKER_BUTTON = 'button[aria-label="Open mode picker"]'
_MODE_LABEL_JS = (
    "() => { const b = document.querySelector("
    "'button[aria-label=\"Open mode picker\"]'"
    "); return b ? b.textContent.trim() : ''; }"
)
_INPUT_SELECTOR = 'div[contenteditable="true"], textarea'
_JSPB_HEADER = "x-goog-ext-525001261-jspb"

_TRACKED_HEADERS: tuple[str, ...] = (
    "x-goog-ext-525001261-jspb",
    "x-goog-ext-73010989-jspb",
    "x-goog-ext-73010990-jspb",
)


def _lookup_model(model_key: str) -> tuple[str, str, str, int] | None:
    """Return the CAPTURE_MODELS entry for a canonical key, or None."""
    for entry in CAPTURE_MODELS:
        if entry[0] == model_key:
            return entry
    return None


async def capture_model_template(
    page: Any, model_key: str
) -> dict | None:
    """Switch to model_key, submit trivial prompt, intercept StreamGenerate.

    Returns ``{"inner": list, "headers": dict[str, str], "raw_jspb": str}``
    on success, or ``None`` on any failure (unknown key, timeout, missing
    intercept, etc.).

    - ``inner``: the parsed ``reference_inner`` list from the intercepted
      request's post_data.
    - ``headers``: the subset of tracked jspb header names that were present
      on the intercepted request.
    - ``raw_jspb``: the raw ``x-goog-ext-525001261-jspb`` header value.
    """
    entry = _lookup_model(model_key)
    if entry is None:
        return None

    _, data_test_id, target_label, _variant = entry

    # Lazy import to avoid a circular dependency: waa_token imports
    # capture_all_models from this module.
    from .waa_token import _parse_stream_generate_request

    captured: dict[str, Any] = {
        "inner": None,
        "headers": {},
        "raw_jspb": None,
    }
    done = asyncio.Event()

    def _extract(request: Any) -> bool:
        if done.is_set():
            return True
        try:
            url = getattr(request, "url", None)
            if isinstance(url, str) and url and "/StreamGenerate" not in url:
                return False
            req_headers = request.headers or {}
            raw_jspb = None
            if hasattr(req_headers, "get"):
                raw_jspb = req_headers.get(
                    _JSPB_HEADER
                ) or req_headers.get(_JSPB_HEADER.lower())
            if not raw_jspb:
                return False
            post_data = getattr(request, "post_data", None)
            parsed = None
            if post_data:
                parsed = _parse_stream_generate_request(post_data)
            reference_inner = (
                parsed.get("reference_inner") if parsed else None
            )
            captured["raw_jspb"] = raw_jspb
            captured["inner"] = reference_inner
            for hname in _TRACKED_HEADERS:
                value = req_headers.get(hname) or req_headers.get(
                    hname.lower()
                )
                if value is not None:
                    captured["headers"][hname] = value
            done.set()
            return True
        except Exception:
            return False

    async def _handler(route: Any) -> None:
        try:
            _extract(route.request)
        finally:
            try:
                await route.abort()
            except Exception:
                pass

    def _observer(request: Any) -> None:
        _extract(request)

    try:
        await page.route("**/StreamGenerate*", _handler)
    except Exception:
        return None
    try:
        page.on("request", _observer)
    except Exception:
        pass

    try:
        current_label = await page.evaluate(_MODE_LABEL_JS)
    except Exception:
        current_label = None

    if current_label != target_label:
        try:
            await page.click(_MODE_PICKER_BUTTON)
            option_selector = f'[data-test-id="bard-mode-option-{data_test_id}"]'
            await page.wait_for_selector(option_selector)
            await page.click(option_selector)
        except Exception:
            return None

    try:
        await page.click(_INPUT_SELECTOR)
        await page.type(_INPUT_SELECTOR, "hi")
        await page.keyboard.press("Enter")
    except Exception:
        return None

    try:
        await asyncio.wait_for(done.wait(), timeout=_CAPTURE_TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        return None

    if captured["raw_jspb"] is None:
        return None
    return captured


async def capture_one_model(page: Any, model_key: str) -> str | None:
    """Thin wrapper around ``capture_model_template`` returning raw_jspb only."""
    # Call through the module so tests can patch ``capture_model_template``
    # at the module level and have this wrapper observe the patch.
    from . import template_capture as _self

    result = await _self.capture_model_template(page, model_key)
    if result is None:
        return None
    return result.get("raw_jspb")


async def capture_all_models(
    page: Any,
    models: Sequence[str] = ("flash", "pro", "thinking"),
    sleep: Callable[[float], Awaitable[None]] | None = None,
    jitter_range: tuple[float, float] = (3.0, 8.0),
    rng: Callable[[float, float], float] | None = None,
) -> dict[str, str]:
    """Capture per-model jspb templates in order with randomized inter-model sleeps."""
    if sleep is None:
        sleep = asyncio.sleep
    if rng is None:
        rng = random.uniform

    result: dict[str, str] = {}
    model_list = list(models)
    for i, key in enumerate(model_list):
        if i > 0:
            delay = rng(*jitter_range)
            await sleep(delay)
        value = await capture_one_model(page, key)
        if value is not None:
            result[key] = value

    return result
