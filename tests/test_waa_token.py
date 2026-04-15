"""
Tests for WAA/BotGuard token harvester.

R1 (complete): Cookie conversion and token dispatch -- tests 1-6.
R2 (current):  Token injection in _generate() -- tests 7-9.

R1 tests cover the plumbing that connects GeminiClient to a WAA token
provider: cookie format conversion, dispatch logic based on provider type,
token format validation, and graceful exception handling.

R2 tests verify that _generate() calls _get_waa_token() and injects the
returned token into inner_req_list[3] of the StreamGenerate request body.
These tests MUST FAIL because _generate() does not yet call _get_waa_token().
"""

import asyncio
import unittest
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import orjson as json
from curl_cffi.requests import Cookies


# ---------------------------------------------------------------------------
# Test 1: Cookie conversion (standalone utility function)
# ---------------------------------------------------------------------------

class TestCookiesForPlaywright(unittest.TestCase):
    """
    _cookies_for_playwright must convert an httpx.Cookies jar into the list-of-
    dicts format that Playwright's browser_context.add_cookies() expects.

    Each dict must have at minimum: name, value, domain, path.
    """

    def test_converts_httpx_cookies_to_playwright_format(self):
        """
        Given an httpx.Cookies jar with two cookies set on .google.com,
        _cookies_for_playwright should return a list of dicts, each containing
        at least {name, value, domain, path}.

        This test fails because the module does not exist yet.
        """
        from gemini_webapi.utils.waa_token import _cookies_for_playwright

        jar = Cookies()
        jar.set("__Secure-1PSID", "sid_value_abc", domain=".google.com")
        jar.set("__Secure-1PSIDTS", "ts_value_xyz", domain=".google.com")

        result = _cookies_for_playwright(jar)

        # Must be a list
        self.assertIsInstance(result, list)
        # Must have exactly 2 entries (one per cookie)
        self.assertEqual(len(result), 2)

        # Each entry must be a dict with the required keys
        required_keys = {"name", "value", "domain", "path"}
        for cookie_dict in result:
            self.assertIsInstance(cookie_dict, dict)
            self.assertTrue(
                required_keys.issubset(cookie_dict.keys()),
                f"Cookie dict missing required keys. "
                f"Expected at least {required_keys}, got {set(cookie_dict.keys())}",
            )

        # Verify actual values are present (order-independent)
        names = {d["name"] for d in result}
        self.assertIn("__Secure-1PSID", names)
        self.assertIn("__Secure-1PSIDTS", names)

        # Find the 1PSID entry and verify its value and domain
        psid_entry = next(d for d in result if d["name"] == "__Secure-1PSID")
        self.assertEqual(psid_entry["value"], "sid_value_abc")
        self.assertEqual(psid_entry["domain"], ".google.com")
        self.assertEqual(psid_entry["path"], "/")


# ---------------------------------------------------------------------------
# Tests 2-5: _get_waa_token dispatch on GeminiClient
# ---------------------------------------------------------------------------

def _make_client_with_provider(waa_token_provider=None):
    """Create a minimal GeminiClient with waa_token_provider set.

    Uses __new__ to skip __init__, then manually sets the attributes
    that _get_waa_token needs. This avoids requiring real credentials
    or network access.
    """
    from gemini_webapi.client import GeminiClient

    client = GeminiClient.__new__(GeminiClient)
    client._running = True
    client.cookies = Cookies()
    client.cookies.set("__Secure-1PSID", "test_sid", domain=".google.com")
    client.verbose = False
    client.waa_token_provider = waa_token_provider
    return client


class TestGetWaaTokenNoneProvider(unittest.IsolatedAsyncioTestCase):
    """
    When waa_token_provider is None (default), _get_waa_token() should
    return None immediately without doing any work. This preserves backward
    compatibility -- existing users who don't set the parameter get the
    current behavior with no WAA token.
    """

    async def test_returns_none_when_provider_is_none(self):
        """
        Given a GeminiClient with waa_token_provider=None,
        _get_waa_token() should return None.

        This test fails because _get_waa_token does not exist yet.
        """
        client = _make_client_with_provider(waa_token_provider=None)

        result = await client._get_waa_token()

        self.assertEqual(result, (None, None))


class TestGetWaaTokenCallableProvider(unittest.IsolatedAsyncioTestCase):
    """
    When waa_token_provider is an async callable, _get_waa_token should
    call it with self.cookies and return its result as the token.
    """

    async def test_calls_callable_provider_with_cookies(self):
        """
        Given a GeminiClient with waa_token_provider set to an async callable
        that returns a valid token string, _get_waa_token() should:
          1. Call the provider with self.cookies as the argument
          2. Return the provider's return value

        This test fails because _get_waa_token does not exist yet.
        """
        fake_token = "!botguard_token_abc123"
        mock_provider = AsyncMock(return_value=fake_token)
        client = _make_client_with_provider(waa_token_provider=mock_provider)

        result = await client._get_waa_token()

        # Provider should have been called exactly once with the client's cookies
        mock_provider.assert_awaited_once_with(client.cookies)
        # Return value should be the token from the provider
        self.assertEqual(result, (fake_token, None))


class TestGetWaaTokenValidatesFormat(unittest.IsolatedAsyncioTestCase):
    """
    WAA/BotGuard tokens always start with '!'. If the provider returns
    something that doesn't match this format, _get_waa_token should
    discard it and return None to avoid injecting garbage into the request.
    """

    async def test_rejects_token_not_starting_with_exclamation(self):
        """
        Given a provider that returns a string NOT starting with '!',
        _get_waa_token() should return None (invalid format).

        This test fails because _get_waa_token does not exist yet.
        """
        mock_provider = AsyncMock(return_value="invalid_no_exclamation_prefix")
        client = _make_client_with_provider(waa_token_provider=mock_provider)

        result = await client._get_waa_token()

        # The provider was called (it's a valid callable)
        mock_provider.assert_awaited_once()
        # But the return value is rejected because it doesn't start with '!'
        self.assertEqual(
            result, (None, None),
            "Token not starting with '!' should be rejected. "
            "_get_waa_token must return (None, None) for invalid token format.",
        )

    async def test_rejects_non_string_token(self):
        """
        Given a provider that returns a non-string value (e.g., an int),
        _get_waa_token() should return None.

        This test fails because _get_waa_token does not exist yet.
        """
        mock_provider = AsyncMock(return_value=42)
        client = _make_client_with_provider(waa_token_provider=mock_provider)

        result = await client._get_waa_token()

        self.assertEqual(
            result, (None, None),
            "Non-string return from provider should be rejected.",
        )


class TestGetWaaTokenExceptionHandling(unittest.IsolatedAsyncioTestCase):
    """
    If the provider raises any exception, _get_waa_token must catch it,
    log a warning, and return None. It must NOT propagate the exception
    upward -- WAA token harvesting is best-effort; failure should degrade
    gracefully to the current 60s stream behavior rather than crash.
    """

    async def test_catches_provider_exception_and_returns_none(self):
        """
        Given a provider that raises RuntimeError, _get_waa_token()
        should catch it and return None without propagating.

        This test fails because _get_waa_token does not exist yet.
        """
        mock_provider = AsyncMock(
            side_effect=RuntimeError("browser crashed during harvest")
        )
        client = _make_client_with_provider(waa_token_provider=mock_provider)

        # Must NOT raise -- the exception should be caught internally
        result = await client._get_waa_token()

        mock_provider.assert_awaited_once()
        self.assertEqual(
            result, (None, None),
            "Provider exception should be caught. _get_waa_token must return "
            "(None, None) on failure, not propagate the exception.",
        )


# ---------------------------------------------------------------------------
# R2: Token injection in _generate()
#
# _generate() builds a 69-element inner_req_list and serialises it into
# the "f.req" form field sent to StreamGenerate.  Position [3] in that
# list must carry the WAA/BotGuard token when a provider is configured,
# and remain None otherwise.
#
# Strategy:
#   1. Build a GeminiClient with all attributes _generate() touches.
#   2. Mock self.client.stream to capture the data= argument and then
#      raise GeminiError (non-retryable) so _generate exits quickly.
#   3. Parse data["f.req"] back into Python to inspect inner_req_list[3].
# ---------------------------------------------------------------------------

def _make_generate_client(waa_token_provider=None):
    """Create a GeminiClient wired up just enough for _generate() to reach
    the self.client.stream() call.

    Skips __init__ via __new__ and manually sets every attribute that
    _generate() (and the @running decorator) access.
    """
    from gemini_webapi.client import GeminiClient

    client = GeminiClient.__new__(GeminiClient)
    # @running decorator checks
    client._running = True
    client.timeout = 300
    client.auto_close = False
    client.close_delay = 300
    client.auto_refresh = True
    client.refresh_interval = 540
    client.verbose = False
    client.watchdog_timeout = 30

    # _generate() reads these directly
    client.cookies = Cookies()
    client.cookies.set("__Secure-1PSID", "test_sid", domain=".google.com")
    client.access_token = "test_at"
    client.build_label = "bl_test"
    client.session_id = "sid_test"
    client._reqid = 10000
    client._lock = asyncio.Lock()
    client.waa_token_provider = waa_token_provider
    client.kwargs = {}

    return client


def _extract_inner_req_list(mock_stream):
    """Parse the captured request data from the mocked stream call and
    return the inner_req_list as a Python list.

    self.client.stream is called as:
        self.client.stream("POST", url, params=..., headers=..., data=request_data)

    request_data["f.req"] is JSON: [None, <json-encoded inner_req_list>]
    """
    # stream() was called once; grab the data= kwarg
    _, kwargs = mock_stream.call_args
    request_data = kwargs["data"]
    f_req = request_data["f.req"]

    # f.req is a JSON string: [null, "<inner_req_list as JSON string>"]
    outer = json.loads(f_req)
    inner_req_list = json.loads(outer[1])
    return inner_req_list


from gemini_webapi.exceptions import GeminiError as _GE


class _StreamAbort(_GE):
    """Sentinel exception raised inside the mocked stream to exit _generate()
    quickly without triggering @running retries.

    Extends GeminiError so the @running decorator does NOT retry.
    """
    pass


def _make_stream_mock():
    """Return an AsyncMock that behaves like httpx.AsyncClient.stream().

    The mock is an async context manager whose __aenter__ returns a
    response-like object with status_code=200 and an aiter_bytes that
    immediately raises _StreamAbort to bail out of _generate() before
    any response processing.
    """
    response = MagicMock()
    response.status_code = 200

    async def _abort_iter():
        raise _StreamAbort("bail out of _generate")
        yield  # pragma: no cover  -- makes this an async generator

    response.aiter_content = _abort_iter

    @asynccontextmanager
    async def _stream_cm(*args, **kwargs):
        yield response

    mock_stream = MagicMock(side_effect=_stream_cm)
    return mock_stream


class TestTokenInjectedIntoGenerateRequest(unittest.IsolatedAsyncioTestCase):
    """
    When _get_waa_token() returns a valid token, _generate() must set
    inner_req_list[3] to that token before serialising the request.

    This test fails because _generate() does not yet call _get_waa_token()
    or assign inner_req_list[3].
    """

    async def test_inner_req_list_3_contains_token_when_provider_returns_valid_token(self):
        """
        Given a GeminiClient whose _get_waa_token returns '!test_token_abc',
        calling _generate() should produce a StreamGenerate request where
        inner_req_list[3] == '!test_token_abc'.

        Why this matters: Without the WAA token at position [3], Google's
        server enforces a 60-second stream timeout during extended thinking.
        """
        from gemini_webapi.constants import Model

        client = _make_generate_client(waa_token_provider=True)
        mock_stream = _make_stream_mock()
        client.client = MagicMock()
        client.client.stream = mock_stream
        client.client.cookies = Cookies()

        expected_token = "!test_token_abc"

        with patch.object(client, "_get_waa_token", new_callable=AsyncMock, return_value=(expected_token, "fakehash123")):
            try:
                async for _ in client._generate(
                    prompt="hello",
                    model=Model.UNSPECIFIED,
                ):
                    pass  # pragma: no cover
            except _StreamAbort:
                pass  # Expected -- our mock aborts the stream

        # The stream mock must have been called exactly once
        mock_stream.assert_called_once()

        inner_req_list = _extract_inner_req_list(mock_stream)

        self.assertEqual(
            inner_req_list[3],
            expected_token,
            f"inner_req_list[3] should be the WAA token '{expected_token}', "
            f"but got {inner_req_list[3]!r}. "
            "_generate() must call _get_waa_token() and inject the result "
            "into position [3] of the request list.",
        )


class TestNoTokenWhenProviderIsNone(unittest.IsolatedAsyncioTestCase):
    """
    When waa_token_provider is None (default), inner_req_list[3] must stay
    None so the request matches the pre-WAA behavior exactly.

    This test should pass with current code (inner_req_list[3] defaults to
    None from the `[None] * 69` initialization), but it documents the
    contract and guards against accidental injection of garbage.
    """

    async def test_inner_req_list_3_is_null_without_provider(self):
        """
        Given a GeminiClient with waa_token_provider=None,
        inner_req_list[3] in the StreamGenerate request must be null.

        Why this matters: Sending a malformed or empty token at position [3]
        could cause Google's server to reject the request entirely, which
        would be worse than the 60-second limit.
        """
        from gemini_webapi.constants import Model

        client = _make_generate_client(waa_token_provider=None)
        mock_stream = _make_stream_mock()
        client.client = MagicMock()
        client.client.stream = mock_stream
        client.client.cookies = Cookies()

        with patch.object(client, "_get_waa_token", new_callable=AsyncMock, return_value=(None, None)):
            try:
                async for _ in client._generate(
                    prompt="hello",
                    model=Model.UNSPECIFIED,
                ):
                    pass  # pragma: no cover
            except _StreamAbort:
                pass  # Expected

        mock_stream.assert_called_once()

        inner_req_list = _extract_inner_req_list(mock_stream)

        self.assertIsNone(
            inner_req_list[3],
            f"inner_req_list[3] should be None when no WAA token provider is "
            f"configured, but got {inner_req_list[3]!r}.",
        )


class TestGenerateCallsGetWaaToken(unittest.IsolatedAsyncioTestCase):
    """
    _generate() must actually invoke _get_waa_token() on every call so that
    fresh tokens are obtained per request.  This is the integration glue
    between the dispatch logic (R1) and the request builder (R2).

    This test fails because _generate() does not yet call _get_waa_token().
    """

    async def test_get_waa_token_is_called_during_generate(self):
        """
        Verify that _generate() calls _get_waa_token() exactly once when
        building the StreamGenerate request.

        Why this matters: If _generate() never calls _get_waa_token(), the
        provider dispatch logic from R1 is dead code -- tokens are harvested
        but never used. This test catches that wiring gap.
        """
        from gemini_webapi.constants import Model

        client = _make_generate_client(waa_token_provider=True)
        mock_stream = _make_stream_mock()
        client.client = MagicMock()
        client.client.stream = mock_stream
        client.client.cookies = Cookies()

        mock_get_waa = AsyncMock(return_value=("!fresh_token_xyz", "hash456"))

        with patch.object(client, "_get_waa_token", mock_get_waa):
            try:
                async for _ in client._generate(
                    prompt="hello",
                    model=Model.UNSPECIFIED,
                ):
                    pass  # pragma: no cover
            except _StreamAbort:
                pass  # Expected

        mock_get_waa.assert_awaited_once()


# ---------------------------------------------------------------------------
# Phase 1: reference_inner_req_list capture
#
# The harvester already extracts inner[3] (WAA token) and inner[4] (BotGuard
# hash) from the intercepted StreamGenerate request.  Phase 1 extends this
# to capture the FULL 80-element inner_req_list so downstream code can diff
# it against what the client is about to send -- catching payload drift
# (e.g. Google silently enforcing inner[67]=0 for Pro on 2026-04-15) on the
# very next request instead of after a manual DevTools capture.
#
# Contract:
#   1. A pure helper `_parse_stream_generate_request(post_data) -> dict|None`
#      exists in utils.waa_token with keys token, botguard_hash, reference_inner.
#   2. `harvest_waa_token` returns a 5-tuple appended with reference_inner.
#   3. `GeminiClient._get_waa_token` unpacks the 5-tuple and stores the
#      reference_inner in `self._reference_inner_req_lists["flash"]`.
#   4. 3-tuple and 4-tuple returns from older harvester versions still work.
# ---------------------------------------------------------------------------

from urllib.parse import urlencode


def _build_post_data(inner_req_list: list) -> str:
    """Build a StreamGenerate-style post_data string wrapping inner_req_list.

    Matches the real wire format the harvester parses:
        f.req = [null, "<json string of inner_req_list>"]
        &at=<anti-csrf>
    """
    outer = [None, json.dumps(inner_req_list).decode("utf-8")]
    f_req = json.dumps(outer).decode("utf-8")
    return urlencode({"f.req": f_req, "at": "fake_anti_csrf_token"})


def _make_reference_inner(token: str, bg_hash: str) -> list:
    """Build a synthetic 80-element inner_req_list with known slot values."""
    inner: list[Any] = [None] * 80
    inner[3] = token
    inner[4] = bg_hash
    inner[67] = 0  # The slot that started the whole saga on 2026-04-15
    inner[0] = "hello"  # Representative prompt slot
    return inner


class TestParseStreamGenerateRequestHappyPath(unittest.TestCase):
    """
    `_parse_stream_generate_request` must decode a real-shape post_data and
    return a dict exposing token, botguard_hash, and the full reference inner
    list.  The reference_inner is the Phase 1 deliverable -- it's what
    downstream payload-drift detection will diff against.
    """

    def test_extracts_token_hash_and_full_inner_list(self):
        """
        Given a post_data containing f.req wrapping an 80-element inner list
        with inner[3]='!tok', inner[4]='aabb...cc', and inner[67]=0, the
        helper returns {token, botguard_hash, reference_inner} where
        reference_inner is the full list with those slots preserved.

        This test fails because `_parse_stream_generate_request` does not
        exist in `gemini_webapi.utils.waa_token`.
        """
        from gemini_webapi.utils.waa_token import _parse_stream_generate_request

        token = "!fake_waa_token_payload_for_test"
        bg_hash = "a" * 32
        inner = _make_reference_inner(token, bg_hash)
        post_data = _build_post_data(inner)

        result = _parse_stream_generate_request(post_data)

        self.assertIsNotNone(result)
        self.assertEqual(result["token"], token)
        self.assertEqual(result["botguard_hash"], bg_hash)

        reference_inner = result["reference_inner"]
        self.assertIsInstance(reference_inner, list)
        self.assertEqual(len(reference_inner), 80)
        self.assertEqual(reference_inner[3], token)
        self.assertEqual(reference_inner[4], bg_hash)
        self.assertEqual(reference_inner[67], 0)
        self.assertEqual(reference_inner[0], "hello")


class TestParseStreamGenerateRequestBadInput(unittest.TestCase):
    """
    The harvester is best-effort -- a malformed capture must degrade to None
    rather than raise, so a single bad intercept cannot crash the browser
    session or leak an exception into the client's token fetch path.
    """

    def test_returns_none_on_missing_f_req(self):
        """
        Given a post_data that has no `f.req` field at all, the helper
        returns None without raising.

        This test fails because `_parse_stream_generate_request` does not
        exist in `gemini_webapi.utils.waa_token`.
        """
        from gemini_webapi.utils.waa_token import _parse_stream_generate_request

        post_data = urlencode({"at": "only_anti_csrf_no_f_req"})

        result = _parse_stream_generate_request(post_data)

        self.assertIsNone(result)


class TestHarvestWaaTokenReturnsReferenceInner(unittest.IsolatedAsyncioTestCase):
    """
    `harvest_waa_token` must return a 5-tuple whose last element is the
    full 80-element reference inner list captured from the intercepted
    StreamGenerate request.

    NOTE ON TESTABILITY: Mocking the full async_playwright chain is
    prohibitively brittle (nested async context managers, Route objects,
    event loops).  Instead, this test drives the pure parser helper and
    asserts the contract that harvest_waa_token is expected to thread its
    output through.  The Green-phase assumption is:

        - `_parse_stream_generate_request` is invoked on every intercepted
          post_data inside `_handle_route`.
        - Its `reference_inner` value is stored alongside token/hash in
          the harvester's closure state.
        - `harvest_waa_token` returns
          (token, browser_version, model_ids, botguard_hash, reference_inner).

    The 5-tuple return shape is covered directly here by inspecting the
    function signature / return-annotation is not practical; instead we
    assert the parser output has the shape the harvester will propagate,
    and we assert via `TestGetWaaTokenStoresReferenceInner` below that
    the client-side unpack handles the 5-tuple correctly.  Together these
    two tests pin down the contract end-to-end without Playwright mocks.
    """

    async def test_parser_output_matches_5tuple_contract(self):
        """
        The parser must expose a reference_inner field that is a plain
        80-element list -- this IS what harvest_waa_token's 5-tuple last
        element will be.  If this assertion fails, harvest_waa_token has
        nothing to put in position [4] of its return tuple.

        This test fails because `_parse_stream_generate_request` does not
        exist yet.
        """
        from gemini_webapi.utils.waa_token import _parse_stream_generate_request

        token = "!another_waa_token"
        bg_hash = "b" * 32
        inner = _make_reference_inner(token, bg_hash)
        post_data = _build_post_data(inner)

        parsed = _parse_stream_generate_request(post_data)

        self.assertIsNotNone(parsed)
        reference_inner = parsed["reference_inner"]
        self.assertEqual(len(reference_inner), 80)
        self.assertEqual(reference_inner[3], token)
        self.assertEqual(reference_inner[4], bg_hash)
        self.assertEqual(reference_inner[67], 0)


class TestGetWaaTokenStoresReferenceInner(unittest.IsolatedAsyncioTestCase):
    """
    When `harvest_waa_token` returns the new 5-tuple, `_get_waa_token` must:
      1. Keep returning the (token, botguard_hash) 2-tuple to its callers
         (backward compat -- _generate() does not care about reference_inner).
      2. Stash the reference_inner list on the client under
         `self._reference_inner_req_lists["flash"]` so payload-drift detection
         can pick it up later.

    `_reference_inner_req_lists` is THE contract for Phase 1 -- it is the
    public (albeit underscore-prefixed) API downstream phases will read.
    """

    async def test_stores_reference_inner_under_flash_key(self):
        """
        Given a patched `harvest_waa_token` that returns a 5-tuple with a
        synthetic 80-element reference_inner, `_get_waa_token` should:
          - return `(token, botguard_hash)` to the caller unchanged, AND
          - set `client._reference_inner_req_lists["flash"]` to the
            reference_inner list.

        This test fails because harvest_waa_token currently returns a
        4-tuple and `_reference_inner_req_lists` does not exist on client.
        """
        token = "!tok_five_tuple_test"
        bg_hash = "c" * 32
        reference_inner = _make_reference_inner(token, bg_hash)
        browser_version = "146.0.7680.178"
        model_ids = {"flash": ["deadbeefdeadbeef"]}

        mock_harvest = AsyncMock(
            return_value=(token, browser_version, model_ids, bg_hash, reference_inner)
        )

        client = _make_client_with_provider(waa_token_provider=True)

        with patch(
            "gemini_webapi.utils.waa_token.harvest_waa_token",
            mock_harvest,
        ):
            result = await client._get_waa_token()

        self.assertEqual(result, (token, bg_hash))

        self.assertTrue(
            hasattr(client, "_reference_inner_req_lists"),
            "_get_waa_token must create self._reference_inner_req_lists "
            "when harvest_waa_token returns a 5-tuple.",
        )
        self.assertIn("flash", client._reference_inner_req_lists)
        self.assertEqual(
            client._reference_inner_req_lists["flash"],
            reference_inner,
        )
        self.assertEqual(len(client._reference_inner_req_lists["flash"]), 80)
        self.assertEqual(client._reference_inner_req_lists["flash"][67], 0)


class TestGetWaaTokenFourTupleBackwardCompat(unittest.IsolatedAsyncioTestCase):
    """
    Older pinned installs of the library may still return the 4-tuple from
    `harvest_waa_token`.  `_get_waa_token` must keep working in that case --
    returning the normal (token, hash) 2-tuple and NOT populating the
    reference cache under "flash" (since there's nothing to cache).
    """

    async def test_four_tuple_return_does_not_populate_reference_cache(self):
        """
        Given a patched harvest_waa_token that still returns a 4-tuple,
        _get_waa_token returns `(token, botguard_hash)` and the reference
        cache either does not exist or has no "flash" entry.

        This test fails today because `_reference_inner_req_lists` is an
        attribute the feature must introduce; once introduced, this test
        also guards the 4-tuple branch against accidentally populating
        the cache with None.
        """
        token = "!tok_four_tuple_test"
        bg_hash = "d" * 32

        mock_harvest = AsyncMock(
            return_value=(token, "146.0.7680.178", {}, bg_hash)
        )

        client = _make_client_with_provider(waa_token_provider=True)

        with patch(
            "gemini_webapi.utils.waa_token.harvest_waa_token",
            mock_harvest,
        ):
            result = await client._get_waa_token()

        self.assertEqual(result, (token, bg_hash))

        reference_cache = getattr(client, "_reference_inner_req_lists", {})
        self.assertNotIn(
            "flash",
            reference_cache,
            "4-tuple return from harvest_waa_token must not populate "
            "_reference_inner_req_lists['flash'] -- there's no reference "
            "list to cache.",
        )


# ---------------------------------------------------------------------------
# Phase 1b + Phase 2: target_model_type routing and drift diff helper
#
# Phase 1b widens `_get_waa_token` so callers can specify WHICH model type
# the captured reference_inner belongs to.  Today's bug that kicked off the
# whole drift-detector initiative was Pro-specific, so a "flash"-only cache
# would miss it entirely.  The `target_model_type` argument lets `_generate`
# pre-select the model before harvesting and stash the reference under the
# matching key.
#
# Phase 2 introduces `_diff_inner_req_list(model_type, built_inner)` on the
# client.  It compares the list the client is about to send against the
# cached reference captured from the real browser and returns a list of
# drift entries describing any mismatches.  The EXCLUDED slots are the
# dynamic per-request fields (message content, WAA token, gem id, etc.)
# that would otherwise swamp the diff with noise.
# ---------------------------------------------------------------------------


class TestGetWaaTokenStoresReferenceUnderTargetModelType(unittest.IsolatedAsyncioTestCase):
    """
    Phase 1b: `_get_waa_token(target_model_type="pro")` must cache the
    reference_inner from a 5-tuple return under the key `"pro"` rather
    than the hardcoded `"flash"` key Phase 1 used.

    Why: Today's drift bug was Pro-specific. A flash-only reference cannot
    catch Pro slot drift. The `target_model_type` argument is how
    `_generate` tells `_get_waa_token` which model reference to cache.
    """

    async def test_stores_reference_inner_under_target_model_type_key(self):
        """
        Given a call to `_get_waa_token(target_model_type="pro")` where
        `harvest_waa_token` returns a 5-tuple, the reference_inner should
        be cached under `client._reference_inner_req_lists["pro"]`, not
        under `"flash"`.

        This test fails because `_get_waa_token` today hardcodes the
        cache key as `"flash"` and does not accept a `target_model_type`
        parameter.
        """
        token = "!tok_target_model_type"
        bg_hash = "e" * 32
        reference_inner = _make_reference_inner(token, bg_hash)
        browser_version = "146.0.7680.178"
        model_ids = {"pro": ["cafebabecafebabe"]}

        mock_harvest = AsyncMock(
            return_value=(token, browser_version, model_ids, bg_hash, reference_inner)
        )

        client = _make_client_with_provider(waa_token_provider=True)
        client._reference_inner_req_lists = {}

        with patch(
            "gemini_webapi.utils.waa_token.harvest_waa_token",
            mock_harvest,
        ):
            result = await client._get_waa_token(target_model_type="pro")

        # Backward compat: return shape unchanged
        self.assertEqual(result, (token, bg_hash))

        # The reference must be cached under "pro", not "flash"
        self.assertIn("pro", client._reference_inner_req_lists)
        self.assertEqual(
            client._reference_inner_req_lists["pro"],
            reference_inner,
        )
        self.assertIsNone(
            client._reference_inner_req_lists.get("flash"),
            "`target_model_type='pro'` must NOT populate the 'flash' key. "
            "Today's Pro drift bug would be silently miscategorised.",
        )


class TestGetWaaTokenDefaultsToFlashKey(unittest.IsolatedAsyncioTestCase):
    """
    Phase 1b backward-compat: callers that don't pass `target_model_type`
    should continue to see the reference cached under `"flash"` so that
    existing Phase 1 tests and any in-flight callers keep working.
    """

    async def test_defaults_to_flash_when_target_model_type_is_none(self):
        """
        Given `_get_waa_token(target_model_type=None)` explicitly passed,
        the reference_inner should still be cached under the legacy
        `"flash"` key so existing callers keep working.

        This test fails because `_get_waa_token` does not yet accept a
        `target_model_type` keyword argument at all -- calling it with
        `target_model_type=None` raises TypeError.
        """
        token = "!tok_default_key"
        bg_hash = "f" * 32
        reference_inner = _make_reference_inner(token, bg_hash)

        mock_harvest = AsyncMock(
            return_value=(token, "146.0.7680.178", {}, bg_hash, reference_inner)
        )

        client = _make_client_with_provider(waa_token_provider=True)
        client._reference_inner_req_lists = {}

        with patch(
            "gemini_webapi.utils.waa_token.harvest_waa_token",
            mock_harvest,
        ):
            await client._get_waa_token(target_model_type=None)

        self.assertEqual(
            client._reference_inner_req_lists.get("flash"),
            reference_inner,
            "Default (target_model_type=None) must fall back to 'flash' key.",
        )


class TestDiffInnerReqListMissingSlot(unittest.TestCase):
    """
    Phase 2: `_diff_inner_req_list(model_type, built_inner)` must report
    slots where the reference has a non-None value but the built list
    has None -- this is exactly today's bug class (Chrome sends a slot
    the client omits).
    """

    def test_reports_missing_slot_when_client_omits_non_none_reference_value(self):
        """
        Given a cached reference where slot 67 is 0 and a built inner
        where slot 67 is None, `_diff_inner_req_list` returns a single
        drift entry for position 67 with kind `missing_in_client`.

        This test fails because `_diff_inner_req_list` does not exist on
        `GeminiClient` yet.
        """
        from gemini_webapi.client import GeminiClient

        reference: list[Any] = [None] * 80
        reference[67] = 0  # Chrome sends this slot
        built: list[Any] = [None] * 80
        # built[67] stays None -- this is the drift

        client = GeminiClient.__new__(GeminiClient)
        client._reference_inner_req_lists = {"pro": reference}

        drift = client._diff_inner_req_list("pro", built)

        self.assertIsInstance(drift, list)
        self.assertEqual(
            len(drift), 1,
            f"Expected exactly one drift entry for slot 67, got {drift!r}",
        )
        entry = drift[0]
        self.assertEqual(entry["position"], 67)
        self.assertEqual(entry["kind"], "missing_in_client")
        self.assertIsNone(entry["client_value"])
        self.assertEqual(entry["chrome_value"], 0)


class TestDiffInnerReqListValueMismatch(unittest.TestCase):
    """
    Phase 2: when both the reference and built lists have values at the
    same position but the values differ, the helper must emit a
    `value_mismatch` drift entry so Green-phase logging can surface it.
    """

    def test_reports_value_mismatch_when_slot_differs(self):
        """
        Given a cached reference where slot 58 is "chrome_value" and a
        built inner where slot 58 is "client_value", `_diff_inner_req_list`
        returns a single drift entry with kind `value_mismatch`.

        This test fails because `_diff_inner_req_list` does not exist yet.
        """
        from gemini_webapi.client import GeminiClient

        reference: list[Any] = [None] * 80
        reference[58] = "chrome_value"
        built: list[Any] = [None] * 80
        built[58] = "client_value"

        client = GeminiClient.__new__(GeminiClient)
        client._reference_inner_req_lists = {"pro": reference}

        drift = client._diff_inner_req_list("pro", built)

        self.assertEqual(len(drift), 1)
        entry = drift[0]
        self.assertEqual(entry["position"], 58)
        self.assertEqual(entry["kind"], "value_mismatch")
        self.assertEqual(entry["client_value"], "client_value")
        self.assertEqual(entry["chrome_value"], "chrome_value")


class TestDiffInnerReqListExcludesDynamicSlots(unittest.TestCase):
    """
    Phase 2: positions listed in the module-level `_DIFF_EXCLUDED_SLOTS`
    constant must be silently skipped by the diff helper.  These are the
    per-request dynamic slots (message content, WAA token, botguard hash,
    gem id, UUID, temporary chat / deep think flags) -- including them
    would produce noise on every single request and mask the rare real
    drift we actually care about.
    """

    def test_dynamic_slots_are_not_reported(self):
        """
        Given a reference and built inner that differ at slot 3 (WAA
        token, excluded) AND at slot 67 (real drift), `_diff_inner_req_list`
        returns exactly one drift entry -- for slot 67, not slot 3.

        Also asserts the exclusion set is named exactly `_DIFF_EXCLUDED_SLOTS`
        in `gemini_webapi.client` and contains at least the documented
        dynamic slots {0, 2, 3, 4, 19, 59} plus TEMPORARY_CHAT_FLAG_INDEX
        and DEEP_THINK_FLAG_INDEX.

        This test fails because neither `_DIFF_EXCLUDED_SLOTS` nor
        `_diff_inner_req_list` exist yet.
        """
        from gemini_webapi import client as client_module
        from gemini_webapi.client import GeminiClient
        from gemini_webapi.constants import (
            DEEP_THINK_FLAG_INDEX,
            TEMPORARY_CHAT_FLAG_INDEX,
        )

        # 1) Contract on the exclusion set itself -- this IS the behavior
        # the test pins down; the whole point is that these slots are excluded.
        self.assertTrue(
            hasattr(client_module, "_DIFF_EXCLUDED_SLOTS"),
            "`_DIFF_EXCLUDED_SLOTS` must be defined at module level in "
            "gemini_webapi.client.",
        )
        excluded = client_module._DIFF_EXCLUDED_SLOTS
        for required_slot in (0, 2, 3, 4, 19, 59):
            self.assertIn(
                required_slot,
                excluded,
                f"Slot {required_slot} must be in _DIFF_EXCLUDED_SLOTS -- "
                f"it is a documented dynamic per-request slot.",
            )
        self.assertIn(TEMPORARY_CHAT_FLAG_INDEX, excluded)
        self.assertIn(DEEP_THINK_FLAG_INDEX, excluded)

        # 2) The helper must actually honour the exclusion set.
        reference: list[Any] = [None] * 80
        reference[3] = "!chrome_waa_token"  # excluded slot, real diff
        reference[67] = 0  # non-excluded slot, real diff

        built: list[Any] = [None] * 80
        built[3] = "!client_waa_token"  # differs, but must be ignored
        # built[67] stays None -- this is the ONE drift we care about

        client = GeminiClient.__new__(GeminiClient)
        client._reference_inner_req_lists = {"pro": reference}

        drift = client._diff_inner_req_list("pro", built)

        self.assertEqual(
            len(drift), 1,
            f"Expected exactly one drift entry (slot 67); slot 3 should be "
            f"silently excluded. Got: {drift!r}",
        )
        self.assertEqual(drift[0]["position"], 67)


class TestDiffInnerReqListNoCachedReference(unittest.TestCase):
    """
    Phase 2: drift detection is best-effort.  If no reference has been
    captured yet for the requested model type (harvester hasn't run,
    failed, or captured a different model), the helper must degrade
    gracefully by returning `[]` -- no warning, no false positives,
    no exception.
    """

    def test_returns_empty_list_when_no_cached_reference_for_model_type(self):
        """
        Given a GeminiClient with an empty `_reference_inner_req_lists`,
        `_diff_inner_req_list("pro", built_inner)` returns `[]`.

        This test fails because `_diff_inner_req_list` does not exist yet.
        """
        from gemini_webapi.client import GeminiClient

        built: list[Any] = [None] * 80
        built[67] = "something"

        client = GeminiClient.__new__(GeminiClient)
        client._reference_inner_req_lists = {}

        drift = client._diff_inner_req_list("pro", built)

        self.assertEqual(
            drift, [],
            "Missing reference data must degrade to an empty drift list -- "
            "no false positives, no exceptions.",
        )


# ---------------------------------------------------------------------------
# Phase 3: PayloadValidationError + diagnose_model probe
#
# Phase 2's automated drift diff catches slot mismatches during normal
# generate_content flow, but it's silent -- it only logs warnings.  Phase 3
# adds an opt-in diagnostic surface:
#
#   1. A new `PayloadValidationError(GeminiError)` exception class that is
#      NOT an `APIError`, so `@running`'s retry loop leaves it alone.
#   2. `GeminiClient.diagnose_model(model_name)` runs a pro+flash probe:
#        - target model fails with the silent-stream signature
#        - AND flash succeeds
#      -> raise PayloadValidationError with a hint to run
#      `python -m gemini_webapi.diag`.
#
# The probe is narrow on purpose: only the specific "target broken, flash
# fine" pattern turns into PayloadValidationError. Other failures propagate
# unchanged so diagnose_model cannot mask unrelated outages.
# ---------------------------------------------------------------------------


class TestPayloadValidationErrorClass(unittest.TestCase):
    """
    PayloadValidationError must be a GeminiError (so `except GeminiError`
    still catches it) but NOT an APIError (so @running won't retry it --
    retrying a payload-drift failure is pointless; the request will keep
    failing until the client code is fixed).
    """

    def test_payload_validation_error_is_gemini_error_not_api_error(self):
        """
        Contract: PayloadValidationError subclasses GeminiError and NOT
        APIError.  This is the whole retry-policy mechanism -- any
        GeminiError subclass that is not an APIError is surfaced to the
        caller immediately by `@running`.
        """
        from gemini_webapi.exceptions import (
            APIError,
            GeminiError,
            PayloadValidationError,
        )

        self.assertTrue(
            issubclass(PayloadValidationError, GeminiError),
            "PayloadValidationError must subclass GeminiError so users' "
            "generic `except GeminiError` clauses still catch it.",
        )
        self.assertFalse(
            issubclass(PayloadValidationError, APIError),
            "PayloadValidationError must NOT subclass APIError -- the "
            "@running decorator retries APIError, and retrying a payload "
            "drift failure would just burn quota.",
        )

    def test_payload_validation_error_preserves_message(self):
        """
        The exception must preserve whatever message the caller passes --
        `diagnose_model` is the caller that crafts the human-readable
        hint (see test_diagnose_model_raises_with_diag_hint below).
        """
        from gemini_webapi.exceptions import PayloadValidationError

        err = PayloadValidationError("custom diagnostic message")
        self.assertIn("custom diagnostic message", str(err))


class TestRunningDecoratorDoesNotRetryPayloadValidationError(
    unittest.IsolatedAsyncioTestCase
):
    """
    The whole reason PayloadValidationError bypasses APIError is so
    `@running(retry=N)` surfaces it on the first attempt -- no sleeps,
    no re-invocations.  This test pins that policy down by wrapping a
    trivial async function with `@running(retry=5)`, having it raise
    PayloadValidationError, and asserting:
      - the exception propagates on the first call, AND
      - `asyncio.sleep` is never awaited (decorator's inter-retry sleep),
      - the wrapped function body runs exactly once.
    """

    async def test_running_decorator_does_not_retry_payload_validation_error(self):
        """
        Given an async function wrapped with `@running(retry=5)` that
        raises PayloadValidationError, the decorator must NOT retry it:
        the wrapped body runs exactly once, `asyncio.sleep` is never
        called, and the exception propagates unchanged.
        """
        from gemini_webapi.exceptions import PayloadValidationError
        from gemini_webapi.utils.decorators import running

        call_count = 0

        @running(retry=5)
        async def _raises_payload_validation(client):
            nonlocal call_count
            call_count += 1
            raise PayloadValidationError("drift detected in slot 67")

        # Minimal client that satisfies the decorator's _running check.
        client = MagicMock()
        client._running = True

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            with self.assertRaises(PayloadValidationError) as cm:
                await _raises_payload_validation(client)

        self.assertIn("drift detected in slot 67", str(cm.exception))
        self.assertEqual(
            call_count, 1,
            f"Wrapped function must run exactly once, ran {call_count} times. "
            "The @running decorator is retrying PayloadValidationError when "
            "it should be letting it propagate immediately.",
        )
        mock_sleep.assert_not_called()


class TestDiagnoseModelRaisesWhenTargetFailsAndFlashSucceeds(
    unittest.IsolatedAsyncioTestCase
):
    """
    `diagnose_model(model_name)` is the opt-in drift probe.  When the
    target model fails with the silent-stream-cut signature ("Stream
    interrupted or truncated.") AND Flash succeeds in the same session,
    the failure cannot be blamed on quota, cookies, or Google outage --
    it's a payload-drift fingerprint.  The probe raises
    PayloadValidationError with a pointer to the CLI diag tool.
    """

    async def test_diagnose_model_raises_with_diag_hint(self):
        """
        Given a client whose generate_content:
          - raises APIError("Stream interrupted or truncated.") when the
            target model is requested, AND
          - returns a successful ModelOutput when Flash is requested,
        `diagnose_model("gemini-3.1-pro")` must raise PayloadValidationError
        whose message includes both the model name (or "pro") and the
        `python -m gemini_webapi.diag` CLI hint.
        """
        from gemini_webapi.client import GeminiClient
        from gemini_webapi.exceptions import APIError, PayloadValidationError

        client = GeminiClient.__new__(GeminiClient)
        client._running = True
        client.verbose = False

        async def _fake_generate(prompt, *, model=None, **_kw):
            model_str = str(model).lower() if model is not None else ""
            if "flash" in model_str:
                result = MagicMock()
                result.text = "ok"
                return result
            raise APIError("Stream interrupted or truncated.")

        with patch.object(
            client, "generate_content", new=AsyncMock(side_effect=_fake_generate)
        ):
            with self.assertRaises(PayloadValidationError) as cm:
                await client.diagnose_model("gemini-3.1-pro")

        msg = str(cm.exception)
        self.assertIn(
            "python -m gemini_webapi.diag",
            msg,
            f"PayloadValidationError message must point users at the diag "
            f"CLI. Got: {msg!r}",
        )
        self.assertTrue(
            "pro" in msg.lower() or "gemini-3.1-pro" in msg,
            f"PayloadValidationError message must identify the failing "
            f"model. Got: {msg!r}",
        )


class TestDiagnoseModelReturnsNoneWhenTargetSucceeds(
    unittest.IsolatedAsyncioTestCase
):
    """
    When the target model generates successfully, there is no drift to
    diagnose -- `diagnose_model` must simply return None.  (It is the
    caller's responsibility to decide whether to log or surface the
    all-clear result.)
    """

    async def test_returns_none_when_target_succeeds(self):
        """
        Given a client whose generate_content succeeds for both Pro and
        Flash, `diagnose_model("gemini-3.1-pro")` returns None and does
        not raise.
        """
        from gemini_webapi.client import GeminiClient

        client = GeminiClient.__new__(GeminiClient)
        client._running = True
        client.verbose = False

        success = MagicMock()
        success.text = "ok"

        mock_generate = AsyncMock(return_value=success)

        with patch.object(client, "generate_content", new=mock_generate):
            result = await client.diagnose_model("gemini-3.1-pro")

        self.assertIsNone(
            result,
            "diagnose_model must return None on the happy path -- there "
            "is nothing to diagnose when the target model works.",
        )


class TestDiagnoseModelPropagatesNonMatchingErrors(
    unittest.IsolatedAsyncioTestCase
):
    """
    `diagnose_model` narrowly catches the silent-stream-cut signature.
    Any OTHER failure (usage limit, temp block, auth issue) must
    propagate unchanged so users aren't misled into thinking they
    have a payload drift when they actually have a quota problem.
    """

    async def test_propagates_usage_limit_exceeded_unchanged(self):
        """
        Given a client whose Pro call raises UsageLimitExceeded,
        `diagnose_model` must re-raise the original exception, NOT
        convert it to PayloadValidationError.
        """
        from gemini_webapi.client import GeminiClient
        from gemini_webapi.exceptions import (
            PayloadValidationError,
            UsageLimitExceeded,
        )

        client = GeminiClient.__new__(GeminiClient)
        client._running = True
        client.verbose = False

        async def _fake_generate(prompt, *, model=None, **_kw):
            raise UsageLimitExceeded("pro daily quota exhausted")

        with patch.object(
            client, "generate_content", new=AsyncMock(side_effect=_fake_generate)
        ):
            with self.assertRaises(UsageLimitExceeded):
                await client.diagnose_model("gemini-3.1-pro")

        # Sanity: UsageLimitExceeded is not a PayloadValidationError subclass,
        # so the assertRaises above is type-meaningful (not accidentally
        # satisfied by a subclass relationship).
        self.assertFalse(
            issubclass(UsageLimitExceeded, PayloadValidationError),
        )


# ---------------------------------------------------------------------------
# R4: Payload drift diag CLI (Phase 4)
#
# Two new surfaces:
#   1. `build_diagnostic_inner_req_list(model, prompt, chat_metadata)` --
#      a module-level pure helper that produces the same 80-element
#      inner_req_list `_generate` would build, with dynamic slots left as
#      sentinels so the diff helper (which already excludes those slots)
#      can compare against a Chrome reference without spawning a real
#      stream.
#   2. `gemini_webapi.diag.main(argv)` -- an on-demand CLI that harvests
#      a real Chrome StreamGenerate reference for a target model, builds
#      the library's inner for the same model, diffs them slot-by-slot,
#      prints the result, and returns 0 on clean / 1 on drift.
#
# These tests MUST FAIL because neither symbol exists yet.
# ---------------------------------------------------------------------------


class TestBuildDiagnosticInnerReqList(unittest.TestCase):
    """
    `build_diagnostic_inner_req_list` extracts the inner_req_list
    construction logic from `_generate` into a pure function so the
    diag CLI (and any future drift tooling) can produce the exact
    80-element list the library would send -- without spawning a
    browser, hitting the network, or instantiating a client.
    """

    def test_returns_80_element_list_with_expected_slots(self):
        """
        For Model.G_3_1_PRO with the default "probe" prompt, the
        builder returns a list matching the structural contract of
        `_generate`: length 80, message list at slot 0, fixed
        browser-parity slots populated, Pro jspb variant at slot 79.
        """
        from gemini_webapi.client import build_diagnostic_inner_req_list
        from gemini_webapi.constants import Model

        result = build_diagnostic_inner_req_list(Model.G_3_1_PRO)

        self.assertEqual(
            len(result),
            80,
            "inner_req_list must be exactly 80 slots to match _generate",
        )

        # Slot 0: 7-element message_content list starting with prompt text.
        self.assertIsInstance(result[0], list)
        self.assertEqual(
            len(result[0]),
            7,
            f"message_content must have 7 elements; got {len(result[0])}",
        )
        self.assertEqual(
            result[0][0],
            "probe",
            "Default prompt text must populate slot 0[0]",
        )

        # Slot 1: language list.
        self.assertEqual(result[1], ["en"])

        # Slot 67: the drift-prone slot we care about in this release.
        self.assertEqual(
            result[67],
            0,
            "Slot 67 must be 0 to match the client's browser-parity "
            "fixed value (this is precisely the slot we want drift "
            "tooling to surface).",
        )

        # Slot 79: Pro variant from jspb header position 11 == 3.
        self.assertEqual(
            result[79],
            3,
            "Pro variant at slot 79 must equal 3 (jspb[11] for Pro)",
        )

        # Slots 3 and 4 are WAA token / botguard hash. The builder
        # must NOT call the real harvester -- these slots should be
        # either None or a clearly-marked sentinel. Accept both.
        token_slot = result[3]
        hash_slot = result[4]
        self.assertTrue(
            token_slot is None or isinstance(token_slot, str),
            f"Slot 3 must be None or a sentinel string; got {token_slot!r}",
        )
        self.assertTrue(
            hash_slot is None or isinstance(hash_slot, str),
            f"Slot 4 must be None or a sentinel string; got {hash_slot!r}",
        )

    def test_respects_custom_prompt_argument(self):
        """
        Passing prompt="hello world" routes the custom prompt into
        slot 0[0] verbatim so CLI users can supply a probe string
        that more closely matches their failing production payload.
        """
        from gemini_webapi.client import build_diagnostic_inner_req_list
        from gemini_webapi.constants import Model

        result = build_diagnostic_inner_req_list(
            Model.G_3_1_PRO, prompt="hello world"
        )

        self.assertEqual(
            result[0][0],
            "hello world",
            "Custom prompt must propagate to slot 0[0]",
        )

    def test_uses_flash_variant_for_flash_model(self):
        """
        Switching to Model.G_3_0_FLASH must change slot 79 to 1
        (Flash jspb[11] variant). This proves the builder is wired
        to the model argument and not hard-coded to Pro.
        """
        from gemini_webapi.client import build_diagnostic_inner_req_list
        from gemini_webapi.constants import Model

        result = build_diagnostic_inner_req_list(Model.G_3_0_FLASH)

        self.assertEqual(
            result[79],
            1,
            "Flash variant at slot 79 must equal 1 (jspb[11] for Flash)",
        )


class TestDiagCliExitCodes(unittest.IsolatedAsyncioTestCase):
    """
    `gemini_webapi.diag.main` is the on-demand CLI entry point.

    Contract:
      - Returns 0 when the harvester's Chrome reference matches
        what `build_diagnostic_inner_req_list` produces for the
        same model (no drift).
      - Returns 1 when the diff is non-empty, and prints the
        drifted slot position to stdout so maintainers can spot
        it in under 30 seconds.

    Tests mock `harvest_waa_token` so no browser spawns and no
    real Google cookies are required.
    """

    async def test_main_returns_0_when_no_drift(self):
        """
        When the mocked harvester returns a reference_inner that
        is identical to what build_diagnostic_inner_req_list
        produces for Pro, `main` must exit 0 and not raise.
        """
        import io

        from gemini_webapi.client import build_diagnostic_inner_req_list
        from gemini_webapi.constants import Model
        from gemini_webapi.diag import main

        # Reference that matches the builder exactly -> zero drift.
        reference_inner = build_diagnostic_inner_req_list(Model.G_3_1_PRO)
        fake_five_tuple = (
            "!fake_token",
            "146.0.0.0",
            {"pro": "abc"},
            "fake_hash",
            reference_inner,
        )

        captured = io.StringIO()
        with patch(
            "gemini_webapi.diag.harvest_waa_token",
            new=AsyncMock(return_value=fake_five_tuple),
        ), patch("sys.stdout", new=captured):
            exit_code = await main(
                ["--model", "pro", "--cookies", "/nonexistent/fake.json", "--no-cdp"]
            )

        self.assertEqual(
            exit_code,
            0,
            f"main must return 0 when reference matches builder "
            f"(no drift). stdout was: {captured.getvalue()!r}",
        )

    async def test_main_returns_1_when_drift_detected(self):
        """
        When the mocked harvester returns a reference_inner that
        differs from the builder at a single slot, `main` must
        return 1 and print the drifted slot number to stdout so
        the maintainer can see which position drifted.
        """
        import io

        from gemini_webapi.client import build_diagnostic_inner_req_list
        from gemini_webapi.constants import Model
        from gemini_webapi.diag import main

        # Clone the builder output and mutate a non-excluded slot so
        # _diff_inner_req_list will flag it. Slot 67 is the obvious
        # candidate -- it's the very slot we expect drift tooling
        # to surface in practice.
        reference_inner = list(build_diagnostic_inner_req_list(Model.G_3_1_PRO))
        reference_inner[67] = 999  # != builder's 0

        fake_five_tuple = (
            "!fake_token",
            "146.0.0.0",
            {"pro": "abc"},
            "fake_hash",
            reference_inner,
        )

        captured = io.StringIO()
        with patch(
            "gemini_webapi.diag.harvest_waa_token",
            new=AsyncMock(return_value=fake_five_tuple),
        ), patch("sys.stdout", new=captured):
            exit_code = await main(
                ["--model", "pro", "--cookies", "/nonexistent/fake.json", "--no-cdp"]
            )

        self.assertEqual(
            exit_code,
            1,
            f"main must return 1 when a slot drifts. "
            f"stdout was: {captured.getvalue()!r}",
        )

        # The drifted slot number must appear in the CLI output so
        # maintainers can identify it at a glance.
        output = captured.getvalue()
        self.assertIn(
            "67",
            output,
            f"CLI output must name the drifted slot (67). "
            f"Got: {output!r}",
        )


if __name__ == "__main__":
    unittest.main()
