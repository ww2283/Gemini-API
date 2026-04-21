"""Tests for gemini_webapi.utils.template_capture -- per-model jspb capture loop.

R4 Red phase: pins the public contract of the not-yet-implemented
``template_capture`` module. These tests must fail at import time until G4
creates ``src/gemini_webapi/utils/template_capture.py``.

Contract under test (from .doc/plans/2026-04-21_per_model_template_capture.md
and R4 slate entry):

- ``CAPTURE_MODELS``    : tuple of (canonical_key, data_test_id, label,
                          variant_int) for Flash/Pro/Thinking in capture order.
- ``capture_one_model`` : switch mode picker, submit a trivial prompt, intercept
                          the outgoing StreamGenerate request, return the raw
                          ``x-goog-ext-525001261-jspb`` header string. Returns
                          None on any failure (no raise).
- ``capture_all_models``: iterate models Flash -> Pro -> Thinking, call
                          ``capture_one_model`` per entry, insert a randomized
                          sleep between models (not before first, not after
                          last). Return dict of canonical_key -> header string
                          with failed models omitted (partial success is valid).

The production module touches only a narrow subset of Playwright's page API
(route/evaluate/click/type/wait_for_selector/keyboard). That narrow surface
is faked here with plain Python so tests never launch a real browser.

Why a FakePage rather than AsyncMock:
- The capture loop drives an ordered dance (evaluate -> click picker -> click
  option -> type -> press enter -> wait for route handler fire). An AsyncMock
  cannot naturally simulate the asynchronous "a request came in" event on
  the same loop while other awaited calls proceed. FakePage exposes
  ``simulate_stream_generate_request(post_data, headers)`` which the test
  schedules as a background task so the handler fires while the capture
  loop is blocked on ``asyncio.wait``.
- Asserting the exact observable outcome (dict values) instead of mock call
  internals aligns with the TDD guidance: test behavior, not mechanics.
"""
from __future__ import annotations

import asyncio
import unittest
from typing import Any, Awaitable, Callable


# ---------------------------------------------------------------------------
# FakePage: minimal async Playwright-page-like surface for tests
# ---------------------------------------------------------------------------


class _FakeRequest:
    """Mimics ``playwright.async_api.Request`` for the narrow surface we use."""

    def __init__(self, post_data: str, headers: dict[str, str]):
        self.post_data = post_data
        self.headers = headers


class _FakeRoute:
    """Mimics ``playwright.async_api.Route``. ``abort``/``continue_`` are no-ops."""

    def __init__(self, request: _FakeRequest):
        self.request = request
        self.aborted = False
        self.continued = False

    async def abort(self) -> None:
        self.aborted = True

    async def continue_(self) -> None:  # pragma: no cover - unused in tests
        self.continued = True


class FakePage:
    """Pluggable async stand-in for ``playwright.async_api.Page``.

    Only the methods that ``capture_one_model`` / ``capture_all_models`` are
    expected to call are modeled. Anything else the production code reaches
    for will raise ``AttributeError`` -- a deliberate test-time tripwire
    against silent surface expansion.

    Per-model behaviour is driven by a queue of callables attached to
    ``self.on_submit``. When the capture loop triggers the "submit" action
    (Enter key press or Send-button click), the next callable in the queue
    runs. A helper (``self.fire_next_intercept``) is provided for use from
    those callables: it invokes the currently registered route handler with
    a canned _FakeRoute, which is how we simulate a StreamGenerate request
    being intercepted.
    """

    def __init__(
        self,
        *,
        current_mode_label: str = "Fast",
        on_submit: list[Callable[["FakePage"], Awaitable[None]]] | None = None,
    ):
        self._route_handler: Callable[[_FakeRoute], Awaitable[None]] | None = None
        self._route_pattern: str | None = None
        self.current_mode_label = current_mode_label
        self.on_submit = list(on_submit) if on_submit else []
        self.clicks: list[tuple[str, dict[str, Any]]] = []
        self.typed: list[tuple[str, str]] = []
        self.wait_calls: list[tuple[str, dict[str, Any]]] = []
        self.evaluate_calls: list[str] = []
        self.keyboard = _FakeKeyboard(self)
        self._submit_index = 0

    # ------------------------------------------------------------------
    # Playwright-facing async surface
    # ------------------------------------------------------------------
    async def route(self, pattern: str, handler) -> None:
        self._route_pattern = pattern
        self._route_handler = handler

    async def evaluate(self, js: str) -> Any:
        self.evaluate_calls.append(js)
        # The capture loop checks current mode label via mode-picker aria-label.
        if "Open mode picker" in js or "mode picker" in js.lower():
            return self.current_mode_label
        return None

    async def click(self, selector: str, **kwargs: Any) -> None:
        self.clicks.append((selector, kwargs))
        # Mode-picker option click -> update stored label so the next
        # "what mode am I in" evaluate returns the new label.
        if "bard-mode-option-" in selector:
            # selector like '[data-test-id="bard-mode-option-pro"]'
            start = selector.index("bard-mode-option-") + len("bard-mode-option-")
            key = selector[start:].split('"')[0].rstrip("]")
            self.current_mode_label = {
                "flash": "Fast",
                "pro": "Pro",
                "thinking": "Thinking",
            }.get(key, self.current_mode_label)
        # Submit via Send button triggers the queued on_submit callback.
        if "Send" in selector or selector.endswith('button[aria-label*="Send"]'):
            await self._drive_submit()

    async def type(self, selector: str, text: str) -> None:
        self.typed.append((selector, text))

    async def wait_for_selector(self, selector: str, **kwargs: Any) -> None:
        self.wait_calls.append((selector, kwargs))

    # ------------------------------------------------------------------
    # Test helpers (not on the real Playwright surface)
    # ------------------------------------------------------------------
    async def _drive_submit(self) -> None:
        if self._submit_index < len(self.on_submit):
            cb = self.on_submit[self._submit_index]
            self._submit_index += 1
            await cb(self)

    async def fire_intercept(
        self, post_data: str, headers: dict[str, str]
    ) -> None:
        """Invoke the registered route handler with a canned request."""
        if self._route_handler is None:
            return
        route = _FakeRoute(_FakeRequest(post_data, headers))
        await self._route_handler(route)


class _FakeKeyboard:
    """Mimics ``page.keyboard`` -- only ``press`` is used by the capture loop."""

    def __init__(self, page: FakePage):
        self._page = page

    async def press(self, key: str) -> None:
        if key == "Enter":
            await self._page._drive_submit()


# ---------------------------------------------------------------------------
# Helpers for building canned StreamGenerate post_data
# ---------------------------------------------------------------------------


def _post_data_for(model_id: str) -> str:
    """Build URL-encoded post_data with a StreamGenerate-shaped ``f.req`` body.

    The capture helper should not care about the inner body -- it lifts the
    ``x-goog-ext-525001261-jspb`` header directly from the intercepted
    request. We still supply a plausible post_data so any future parsing
    step inside capture_one_model has something to chew on. The interesting
    value is ``inner[4]`` (hex model id) which matches the slot [4] of the
    jspb header we return alongside.
    """
    import urllib.parse

    inner = [
        "hi",
        0,
        None,
        [[], None, None, [[], []], [], [], None, [0]],
        [None, None, None, None, None, None, [1]],
        None,
        0,
        None,
        [1],
        0,
        None,
        None,
        1,
    ]
    inner[4] = model_id  # placeholder slot; not load-bearing for R4 tests
    outer = [None, __import__("json").dumps(inner)]
    f_req = __import__("json").dumps(outer)
    return urllib.parse.urlencode({"f.req": f_req})


def _jspb_header_for(model_key: str) -> str:
    """Return the R4 slate's canonical per-model jspb template string."""
    return {
        "pro": '[1,null,null,null,"797f3d0293f288ad",null,null,0,[4],null,null,3,null,null,3,1]',
        "flash": '[1,null,null,null,"56fdd199312815e2",null,null,0,[4],null,null,3,null,null,1]',
        "thinking": '[1,null,null,null,"aaaaaaaaaaaaaaaa",null,null,0,[4],null,null,3,null,null,2]',
    }[model_key]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCaptureAllModelsReturnsPerModelTemplates(
    unittest.IsolatedAsyncioTestCase
):
    """Happy path: three successful captures, dict has all three keys."""

    async def test_capture_all_models_returns_per_model_templates(self):
        """capture_all_models returns canonical key -> raw jspb for each model.

        The capture loop iterates ``CAPTURE_MODELS`` in order. For each model,
        a submit action fires a simulated StreamGenerate request with a
        distinct ``x-goog-ext-525001261-jspb`` header. The returned dict must
        carry those exact strings, keyed by canonical model name.
        """
        from gemini_webapi.utils.template_capture import capture_all_models

        async def fire_flash(page: FakePage) -> None:
            await page.fire_intercept(
                _post_data_for("56fdd199312815e2"),
                {"x-goog-ext-525001261-jspb": _jspb_header_for("flash")},
            )

        async def fire_pro(page: FakePage) -> None:
            await page.fire_intercept(
                _post_data_for("797f3d0293f288ad"),
                {"x-goog-ext-525001261-jspb": _jspb_header_for("pro")},
            )

        async def fire_thinking(page: FakePage) -> None:
            await page.fire_intercept(
                _post_data_for("aaaaaaaaaaaaaaaa"),
                {"x-goog-ext-525001261-jspb": _jspb_header_for("thinking")},
            )

        page = FakePage(
            current_mode_label="Fast",
            on_submit=[fire_flash, fire_pro, fire_thinking],
        )

        result = await capture_all_models(
            page,
            sleep=lambda s: asyncio.sleep(0),  # no real sleep in tests
            rng=lambda lo, hi: 0.0,
        )

        self.assertEqual(
            set(result.keys()),
            {"flash", "pro", "thinking"},
            f"Expected all three canonical keys, got {set(result.keys())}",
        )
        self.assertEqual(result["flash"], _jspb_header_for("flash"))
        self.assertEqual(result["pro"], _jspb_header_for("pro"))
        self.assertEqual(result["thinking"], _jspb_header_for("thinking"))


class TestCaptureAllModelsUsesInjectedSleepAndRng(
    unittest.IsolatedAsyncioTestCase
):
    """Jitter plumbing: rng produces duration, sleep consumes it, N-1 times."""

    async def test_capture_all_models_uses_injected_sleep_and_rng(self):
        """rng is called with jitter_range; sleep is called with rng's result.

        For N successfully-captured models, there are exactly N-1 inter-model
        sleeps (not before first, not after last). rng must receive the
        configured jitter_range as its two positional args. Sleep must be
        called with whatever rng returned -- this verifies the randomized
        delay is actually threaded through, not recomputed inside sleep.
        """
        from gemini_webapi.utils.template_capture import capture_all_models

        sleep_calls: list[float] = []
        rng_calls: list[tuple[float, float]] = []

        async def record_sleep(seconds: float) -> None:
            sleep_calls.append(seconds)

        # Each rng invocation returns a distinct sentinel so we can verify
        # the sleep received exactly that value (not, e.g., re-randomized).
        rng_return_values = iter([4.25, 6.5, 99.0])

        def record_rng(lo: float, hi: float) -> float:
            rng_calls.append((lo, hi))
            return next(rng_return_values)

        async def fire_flash(p: FakePage) -> None:
            await p.fire_intercept(
                _post_data_for("56fdd199312815e2"),
                {"x-goog-ext-525001261-jspb": _jspb_header_for("flash")},
            )

        async def fire_pro(p: FakePage) -> None:
            await p.fire_intercept(
                _post_data_for("797f3d0293f288ad"),
                {"x-goog-ext-525001261-jspb": _jspb_header_for("pro")},
            )

        async def fire_thinking(p: FakePage) -> None:
            await p.fire_intercept(
                _post_data_for("aaaaaaaaaaaaaaaa"),
                {"x-goog-ext-525001261-jspb": _jspb_header_for("thinking")},
            )

        page = FakePage(
            current_mode_label="Fast",
            on_submit=[fire_flash, fire_pro, fire_thinking],
        )

        result = await capture_all_models(
            page,
            sleep=record_sleep,
            jitter_range=(3.0, 8.0),
            rng=record_rng,
        )

        self.assertEqual(len(result), 3, "All three captures should succeed")
        # N-1 sleeps, N-1 rng calls for N=3 successful captures.
        self.assertEqual(len(rng_calls), 2, f"Expected 2 rng calls, got {rng_calls}")
        self.assertEqual(len(sleep_calls), 2, f"Expected 2 sleeps, got {sleep_calls}")
        # rng must be called with the configured jitter_range each time.
        for lo, hi in rng_calls:
            self.assertEqual((lo, hi), (3.0, 8.0))
        # sleep values must match rng's first two return values verbatim
        # (not re-randomized or replaced with constants).
        self.assertEqual(sleep_calls, [4.25, 6.5])


class TestCaptureOneModelReturnsNoneOnInterceptTimeout(
    unittest.IsolatedAsyncioTestCase
):
    """Intercept never fires -> capture_one_model must return None, not hang."""

    async def test_capture_one_model_returns_none_on_intercept_timeout(self):
        """If no StreamGenerate request fires, capture_one_model returns None.

        The internal wait must be bounded so that flaky Chrome instances or
        rate-limited mode switches can't hang the entire harvester. We
        enforce that by wrapping the call in an outer asyncio.wait_for with
        a short ceiling -- if capture_one_model has no internal timeout it
        will blow past this and raise TimeoutError, failing the test.
        """
        from gemini_webapi.utils.template_capture import capture_one_model

        # on_submit queue is empty -> submit actions are no-ops -> no
        # intercept ever fires. capture_one_model must still return.
        page = FakePage(current_mode_label="Fast", on_submit=[])

        result = await asyncio.wait_for(
            capture_one_model(page, "flash"),
            timeout=20.0,  # generous outer guard; inner timeout should be much tighter
        )

        self.assertIsNone(
            result,
            f"Expected None on intercept timeout, got {result!r}",
        )


class TestCaptureAllModelsPartialSuccess(unittest.IsolatedAsyncioTestCase):
    """Pro capture fails silently; dict has only the two that succeeded."""

    async def test_capture_all_models_partial_success(self):
        """One failed capture must not poison the others.

        Flash and Thinking fire their intercepts; Pro's submit callback is a
        no-op so its intercept never fires. The returned dict must contain
        flash and thinking entries, and Pro must be absent (not present with
        a None value). No exception is raised -- partial success is valid.
        """
        from gemini_webapi.utils.template_capture import capture_all_models

        async def fire_flash(p: FakePage) -> None:
            await p.fire_intercept(
                _post_data_for("56fdd199312815e2"),
                {"x-goog-ext-525001261-jspb": _jspb_header_for("flash")},
            )

        async def pro_no_intercept(p: FakePage) -> None:
            # Submit action ran but intercept never fires.
            return None

        async def fire_thinking(p: FakePage) -> None:
            await p.fire_intercept(
                _post_data_for("aaaaaaaaaaaaaaaa"),
                {"x-goog-ext-525001261-jspb": _jspb_header_for("thinking")},
            )

        page = FakePage(
            current_mode_label="Fast",
            on_submit=[fire_flash, pro_no_intercept, fire_thinking],
        )

        result = await capture_all_models(
            page,
            sleep=lambda s: asyncio.sleep(0),
            rng=lambda lo, hi: 0.0,
        )

        self.assertEqual(
            set(result.keys()),
            {"flash", "thinking"},
            f"Pro should be absent on capture failure; got keys {set(result.keys())}",
        )
        self.assertNotIn(
            "pro",
            result,
            "Failed models must be omitted entirely, not mapped to None",
        )
        self.assertEqual(result["flash"], _jspb_header_for("flash"))
        self.assertEqual(result["thinking"], _jspb_header_for("thinking"))


class TestCaptureOneModelRejectsUnknownKey(unittest.IsolatedAsyncioTestCase):
    """Guard against callers requesting a model the harvester can't handle."""

    async def test_capture_one_model_rejects_unknown_key(self):
        """Unknown canonical key -> return None immediately, don't touch page.

        An unknown key would otherwise send us to a nonexistent mode-picker
        option selector and hang until timeout. Fail fast instead: return
        None without calling route/evaluate/click/type/wait. That keeps the
        failure visible and cheap.
        """
        from gemini_webapi.utils.template_capture import capture_one_model

        page = FakePage(current_mode_label="Fast", on_submit=[])

        result = await capture_one_model(page, "nonexistent")

        self.assertIsNone(result)
        # The page must not have been driven at all -- every interaction
        # would waste time on a doomed capture attempt.
        self.assertEqual(page.clicks, [], "Unknown key must not trigger clicks")
        self.assertEqual(page.typed, [], "Unknown key must not trigger typing")
        self.assertEqual(
            page.evaluate_calls, [], "Unknown key must not trigger JS evaluation"
        )
        self.assertEqual(
            page.wait_calls, [], "Unknown key must not trigger wait_for_selector"
        )


# ---------------------------------------------------------------------------
# R6: capture_model_template shared helper (diag + harvester reuse)
# ---------------------------------------------------------------------------
#
# R6 motivation: diag.py currently has a private ``_capture_via_cdp`` body
# (lines 135-257) that duplicates mode-switch + submit + intercept logic that
# is now in template_capture. We extract a broader ``capture_model_template``
# helper that returns not just the raw jspb header string but also the parsed
# ``inner`` request list and the full tracked-headers dict -- the three
# pieces diag's payload drift diff needs.
#
# Contract under test:
#
# ``capture_model_template(page, model_key) -> dict | None``
# Returns ``{"inner": list, "headers": dict[str, str], "raw_jspb": str}``
# on success, or ``None`` if capture fails (no intercept, timeout, unknown
# key, etc.). ``headers`` contains only tracked jspb header names that
# were actually present on the captured request.
#
# ``capture_one_model(page, model_key) -> str | None`` becomes a thin wrapper
# that delegates to ``capture_model_template`` and returns ``raw_jspb`` only.
# This preserves the G4 contract (and the existing R4 tests above) while
# letting diag access the richer capture result.


def _post_data_with_waa_token(model_id: str) -> str:
    """Build post_data whose ``inner[3]`` starts with ``!``.

    The shared helper (like diag's existing ``handle_route``) filters on
    ``parsed['token'].startswith('!')`` to ignore non-WAA-armed probe
    requests that Chrome fires during initial load. This helper produces
    canonical post_data whose WAA-token slot actually passes that check so
    tests can drive the happy path.
    """
    import urllib.parse

    inner = [
        "hi",
        0,
        None,
        "!abc123waa_token_sentinel",  # inner[3]: WAA token (must start with '!')
        "botguard_hash_sentinel",     # inner[4]: BotGuard hash
        None,
        0,
        None,
        [1],
        0,
        None,
        None,
        1,
    ]
    outer = [None, __import__("json").dumps(inner)]
    f_req = __import__("json").dumps(outer)
    return urllib.parse.urlencode({"f.req": f_req})


class TestCaptureModelTemplateReturnsInnerHeadersAndRawJspb(
    unittest.IsolatedAsyncioTestCase
):
    """capture_model_template returns the full (inner, headers, raw_jspb) bundle."""

    async def test_capture_model_template_returns_inner_headers_and_raw_jspb(self):
        """capture_model_template returns a dict with inner, headers, raw_jspb.

        The helper is the shared capture primitive reused by both diag.py
        (which needs ``inner`` + all tracked headers for the slot diff) and
        by ``capture_one_model`` (which needs only ``raw_jspb``). The result
        dict must carry all three fields populated from the same intercepted
        request so downstream consumers don't have to make separate calls.
        """
        from gemini_webapi.utils.template_capture import capture_model_template

        jspb_value = _jspb_header_for("flash")
        other_jspb_1 = '[999,"other-header-1"]'
        other_jspb_2 = '[999,"other-header-2"]'

        async def fire_flash(p: FakePage) -> None:
            await p.fire_intercept(
                _post_data_with_waa_token("56fdd199312815e2"),
                {
                    "x-goog-ext-525001261-jspb": jspb_value,
                    "x-goog-ext-73010989-jspb": other_jspb_1,
                    "x-goog-ext-73010990-jspb": other_jspb_2,
                },
            )

        page = FakePage(current_mode_label="Fast", on_submit=[fire_flash])

        result = await capture_model_template(page, "flash")

        self.assertIsNotNone(
            result,
            "capture_model_template must not return None on successful capture",
        )
        # Three fields present
        self.assertIn("inner", result)
        self.assertIn("headers", result)
        self.assertIn("raw_jspb", result)
        # raw_jspb is the header string verbatim
        self.assertEqual(result["raw_jspb"], jspb_value)
        # inner is the parsed reference_inner list (from post_data f.req)
        self.assertIsInstance(result["inner"], list)
        self.assertEqual(
            result["inner"][3],
            "!abc123waa_token_sentinel",
            "inner[3] must be the WAA token parsed from post_data",
        )
        self.assertEqual(
            result["inner"][4],
            "botguard_hash_sentinel",
            "inner[4] must be the BotGuard hash parsed from post_data",
        )
        # headers dict contains all tracked jspb header names
        self.assertIsInstance(result["headers"], dict)
        self.assertEqual(
            result["headers"].get("x-goog-ext-525001261-jspb"), jspb_value
        )
        self.assertEqual(
            result["headers"].get("x-goog-ext-73010989-jspb"), other_jspb_1
        )
        self.assertEqual(
            result["headers"].get("x-goog-ext-73010990-jspb"), other_jspb_2
        )


class TestCaptureModelTemplateReturnsNoneOnInterceptTimeout(
    unittest.IsolatedAsyncioTestCase
):
    """capture_model_template must return None (not hang) when no intercept fires."""

    async def test_capture_model_template_returns_none_on_intercept_timeout(self):
        """Bounded wait: no StreamGenerate request -> None within time budget.

        Same rationale as ``test_capture_one_model_returns_none_on_intercept_timeout``:
        a flaky Chrome instance or rate-limited mode switch must not be able
        to hang the entire harvester. The outer ``asyncio.wait_for`` is a
        generous guard; the production helper's internal timeout should be
        much tighter and must trip first, yielding None.
        """
        from gemini_webapi.utils.template_capture import capture_model_template

        # Empty on_submit queue -> submits are no-ops -> intercept never fires.
        page = FakePage(current_mode_label="Fast", on_submit=[])

        result = await asyncio.wait_for(
            capture_model_template(page, "flash"),
            timeout=20.0,
        )

        self.assertIsNone(
            result,
            f"Expected None on intercept timeout, got {result!r}",
        )


class TestCaptureOneModelDelegatesToCaptureModelTemplate(
    unittest.IsolatedAsyncioTestCase
):
    """capture_one_model becomes a thin wrapper around capture_model_template."""

    async def test_capture_one_model_delegates_to_capture_model_template(self):
        """capture_one_model returns raw_jspb from capture_model_template's dict.

        This pins the G6 refactor: ``capture_one_model`` no longer owns the
        capture body; it delegates to ``capture_model_template`` and projects
        out the ``raw_jspb`` field. We verify by patching the shared helper
        to return a stub dict and checking the wrapper extracts the right
        key. This keeps the G4-era R4 contract intact while letting diag
        reach for the richer capture result via the same code path.
        """
        from unittest.mock import patch

        from gemini_webapi.utils import template_capture
        from gemini_webapi.utils.template_capture import capture_one_model

        stub_result = {
            "inner": ["stub-inner"],
            "headers": {"x-goog-ext-525001261-jspb": "stub-jspb-value"},
            "raw_jspb": "stub-jspb-value",
        }

        async def stub_capture_model_template(page, model_key):
            # Verify the wrapper forwarded its arguments unchanged.
            self.assertEqual(model_key, "pro")
            return stub_result

        page = FakePage(current_mode_label="Fast", on_submit=[])

        with patch.object(
            template_capture,
            "capture_model_template",
            new=stub_capture_model_template,
        ):
            returned = await capture_one_model(page, "pro")

        self.assertEqual(
            returned,
            "stub-jspb-value",
            "capture_one_model must return the raw_jspb field of the "
            "capture_model_template result, not the full dict",
        )

    async def test_capture_one_model_returns_none_when_template_helper_returns_none(
        self,
    ):
        """If capture_model_template returns None, capture_one_model also None.

        The wrapper must not blow up indexing into a None result; it must
        propagate the None cleanly so existing capture_all_models partial-
        success handling (which checks ``value is not None``) still works
        unchanged.
        """
        from unittest.mock import patch

        from gemini_webapi.utils import template_capture
        from gemini_webapi.utils.template_capture import capture_one_model

        async def stub_returns_none(page, model_key):
            return None

        page = FakePage(current_mode_label="Fast", on_submit=[])

        with patch.object(
            template_capture,
            "capture_model_template",
            new=stub_returns_none,
        ):
            returned = await capture_one_model(page, "flash")

        self.assertIsNone(returned)


if __name__ == "__main__":
    unittest.main()
