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
from unittest.mock import AsyncMock, MagicMock, patch

import orjson as json
from httpx import Cookies


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

        self.assertIsNone(result)


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
        self.assertEqual(result, fake_token)


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
        self.assertIsNone(
            result,
            "Token not starting with '!' should be rejected. "
            "_get_waa_token must return None for invalid token format.",
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

        self.assertIsNone(
            result,
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
        self.assertIsNone(
            result,
            "Provider exception should be caught. _get_waa_token must return "
            "None on failure, not propagate the exception.",
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

    response.aiter_bytes = _abort_iter

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

        with patch.object(client, "_get_waa_token", new_callable=AsyncMock, return_value=expected_token):
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

        with patch.object(client, "_get_waa_token", new_callable=AsyncMock, return_value=None):
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

        mock_get_waa = AsyncMock(return_value="!fresh_token_xyz")

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


if __name__ == "__main__":
    unittest.main()
