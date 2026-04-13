"""
Tests for deep_think mode in _generate().

R1: Flag injection and array size -- tests 1-3.

Deep think mode is enabled by setting inner_req_list[49] = 20 in the
StreamGenerate request body.  When deep_think=True is passed to _generate(),
the flag must be injected; when deep_think is False or omitted, position [49]
must remain None.  Additionally, the inner_req_list must be extended to 80
elements (up from 69) so that higher-index slots (like [49]) are accessible
without off-by-one confusion and future slots up to [79] are available.

These tests MUST FAIL because:
  - The constants DEEP_THINK_FLAG_INDEX and DEEP_THINK_FLAG_VALUE do not exist.
  - _generate() does not accept a deep_think parameter.
  - inner_req_list is currently 69 elements, not 80.
"""

import asyncio
import unittest
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import orjson as json
from curl_cffi.requests import Cookies

from gemini_webapi.exceptions import GeminiError as _GE


# ---------------------------------------------------------------------------
# Shared test helpers (same pattern as test_waa_token.py)
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


class _StreamAbort(_GE):
    """Sentinel exception raised inside the mocked stream to exit _generate()
    quickly without triggering @running retries.

    Extends GeminiError so the @running decorator does NOT retry.
    """
    pass


def _make_stream_mock():
    """Return a mock that behaves like httpx.AsyncClient.stream().

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


def _extract_inner_req_list(mock_stream):
    """Parse the captured request data from the mocked stream call and
    return the inner_req_list as a Python list.

    self.client.stream is called as:
        self.client.stream("POST", url, params=..., headers=..., data=request_data)

    request_data["f.req"] is JSON: [null, "<inner_req_list as JSON string>"]
    """
    _, kwargs = mock_stream.call_args
    request_data = kwargs["data"]
    f_req = request_data["f.req"]

    outer = json.loads(f_req)
    inner_req_list = json.loads(outer[1])
    return inner_req_list


# ---------------------------------------------------------------------------
# Test 1: deep_think flag is injected when deep_think=True
# ---------------------------------------------------------------------------

class TestDeepThinkFlagInjected(unittest.IsolatedAsyncioTestCase):
    """
    When deep_think=True is passed to _generate(), inner_req_list[49] must
    be set to 20. This is the server-side signal that enables extended
    reasoning / deep think mode.

    This test fails because:
      - DEEP_THINK_FLAG_INDEX and DEEP_THINK_FLAG_VALUE don't exist in constants
      - _generate() does not accept a deep_think parameter
    """

    async def test_deep_think_flag_injected(self):
        """
        Given deep_think=True passed to _generate(), the StreamGenerate
        request body must contain inner_req_list[49] == 20.

        Why this matters: Without this flag at position [49], the server
        does not activate deep think mode, and the response will be a
        standard (non-extended-reasoning) generation.
        """
        from gemini_webapi.constants import (
            DEEP_THINK_FLAG_INDEX,
            DEEP_THINK_FLAG_VALUE,
            Model,
        )

        client = _make_generate_client()
        mock_stream = _make_stream_mock()
        client.client = MagicMock()
        client.client.stream = mock_stream
        client.client.cookies = Cookies()

        # Stub _get_waa_token so the WAA path doesn't interfere
        with patch.object(
            client, "_get_waa_token", new_callable=AsyncMock, return_value=None
        ):
            try:
                async for _ in client._generate(
                    prompt="Explain quantum computing in depth",
                    model=Model.UNSPECIFIED,
                    deep_think=True,
                ):
                    pass  # pragma: no cover
            except _StreamAbort:
                pass  # Expected -- our mock aborts the stream

        mock_stream.assert_called_once()
        inner_req_list = _extract_inner_req_list(mock_stream)

        # Verify the flag index and value match the constants
        self.assertEqual(DEEP_THINK_FLAG_INDEX, 49)
        self.assertEqual(DEEP_THINK_FLAG_VALUE, 20)

        self.assertEqual(
            inner_req_list[DEEP_THINK_FLAG_INDEX],
            DEEP_THINK_FLAG_VALUE,
            f"inner_req_list[{DEEP_THINK_FLAG_INDEX}] should be "
            f"{DEEP_THINK_FLAG_VALUE} when deep_think=True, but got "
            f"{inner_req_list[DEEP_THINK_FLAG_INDEX]!r}. "
            "_generate() must set inner_req_list[49] = 20 when deep_think "
            "is enabled.",
        )


# ---------------------------------------------------------------------------
# Test 2: deep_think flag is NOT set by default
# ---------------------------------------------------------------------------

class TestDeepThinkFlagNotSetByDefault(unittest.IsolatedAsyncioTestCase):
    """
    When deep_think is not passed (defaults to False), inner_req_list[49]
    must remain None. This ensures backward compatibility -- existing callers
    who don't use deep think get the same request structure as before.

    This test fails because:
      - DEEP_THINK_FLAG_INDEX doesn't exist in constants
      - _generate() does not accept a deep_think parameter (though the
        default-False semantics mean the array value *might* be None anyway
        once the array is extended to 80 -- the import failure still blocks)
    """

    async def test_deep_think_flag_not_set_by_default(self):
        """
        Given a _generate() call with no deep_think argument (or
        deep_think=False), inner_req_list[49] must be None.

        Why this matters: Accidentally sending the deep think flag could
        alter response behavior for users who didn't request it, potentially
        causing longer response times or different output formatting.
        """
        from gemini_webapi.constants import DEEP_THINK_FLAG_INDEX, Model

        client = _make_generate_client()
        mock_stream = _make_stream_mock()
        client.client = MagicMock()
        client.client.stream = mock_stream
        client.client.cookies = Cookies()

        with patch.object(
            client, "_get_waa_token", new_callable=AsyncMock, return_value=None
        ):
            try:
                async for _ in client._generate(
                    prompt="Hello",
                    model=Model.UNSPECIFIED,
                    # deep_think not passed -- should default to False
                ):
                    pass  # pragma: no cover
            except _StreamAbort:
                pass  # Expected

        mock_stream.assert_called_once()
        inner_req_list = _extract_inner_req_list(mock_stream)

        self.assertIsNone(
            inner_req_list[DEEP_THINK_FLAG_INDEX],
            f"inner_req_list[{DEEP_THINK_FLAG_INDEX}] should be None when "
            f"deep_think is not enabled, but got "
            f"{inner_req_list[DEEP_THINK_FLAG_INDEX]!r}. "
            "The deep think flag must only be set when explicitly requested.",
        )

    async def test_deep_think_false_explicitly(self):
        """
        Given deep_think=False passed explicitly to _generate(),
        inner_req_list[49] must be None -- same as the default case.

        Why this matters: Explicit False must behave identically to omission.
        """
        from gemini_webapi.constants import DEEP_THINK_FLAG_INDEX, Model

        client = _make_generate_client()
        mock_stream = _make_stream_mock()
        client.client = MagicMock()
        client.client.stream = mock_stream
        client.client.cookies = Cookies()

        with patch.object(
            client, "_get_waa_token", new_callable=AsyncMock, return_value=None
        ):
            try:
                async for _ in client._generate(
                    prompt="Hello",
                    model=Model.UNSPECIFIED,
                    deep_think=False,
                ):
                    pass  # pragma: no cover
            except _StreamAbort:
                pass  # Expected

        mock_stream.assert_called_once()
        inner_req_list = _extract_inner_req_list(mock_stream)

        self.assertIsNone(
            inner_req_list[DEEP_THINK_FLAG_INDEX],
            f"inner_req_list[{DEEP_THINK_FLAG_INDEX}] should be None when "
            f"deep_think=False, but got "
            f"{inner_req_list[DEEP_THINK_FLAG_INDEX]!r}.",
        )


# ---------------------------------------------------------------------------
# Test 3: inner_req_list is 80 elements long
# ---------------------------------------------------------------------------

class TestInnerReqListLength(unittest.IsolatedAsyncioTestCase):
    """
    The inner_req_list must be exactly 80 elements long (extended from 69).
    This is required because the deep think flag lives at index 49 (which fits
    in 69 elements), but the array extension to 80 was observed in Chrome
    network traces when deep think mode is active, and future features may
    use slots up to [79].

    This test fails because inner_req_list is currently initialized as
    [None] * 69 (line 701 of client.py).
    """

    async def test_inner_req_list_length_is_80(self):
        """
        The StreamGenerate request's inner_req_list must have exactly 80
        elements, regardless of whether deep_think is enabled.

        Why this matters: Google's server expects a fixed-size array. If
        the array is too short, unset higher-index slots are missing from
        the serialized JSON, which could cause server-side parsing errors
        or silent feature deactivation.
        """
        from gemini_webapi.constants import Model

        client = _make_generate_client()
        mock_stream = _make_stream_mock()
        client.client = MagicMock()
        client.client.stream = mock_stream
        client.client.cookies = Cookies()

        with patch.object(
            client, "_get_waa_token", new_callable=AsyncMock, return_value=None
        ):
            try:
                async for _ in client._generate(
                    prompt="Hello",
                    model=Model.UNSPECIFIED,
                ):
                    pass  # pragma: no cover
            except _StreamAbort:
                pass  # Expected

        mock_stream.assert_called_once()
        inner_req_list = _extract_inner_req_list(mock_stream)

        self.assertEqual(
            len(inner_req_list),
            80,
            f"inner_req_list should have 80 elements but has "
            f"{len(inner_req_list)}. The array must be extended from "
            f"[None] * 69 to [None] * 80 to accommodate deep think and "
            f"future feature flags.",
        )

    async def test_inner_req_list_length_is_80_with_deep_think(self):
        """
        When deep_think=True, inner_req_list must still be exactly 80
        elements -- the flag injection must not change the array length.

        Why this matters: Ensures the flag is set in-place within the
        pre-allocated array rather than appending or extending it.
        """
        from gemini_webapi.constants import Model

        client = _make_generate_client()
        mock_stream = _make_stream_mock()
        client.client = MagicMock()
        client.client.stream = mock_stream
        client.client.cookies = Cookies()

        with patch.object(
            client, "_get_waa_token", new_callable=AsyncMock, return_value=None
        ):
            try:
                async for _ in client._generate(
                    prompt="Explain deep think",
                    model=Model.UNSPECIFIED,
                    deep_think=True,
                ):
                    pass  # pragma: no cover
            except _StreamAbort:
                pass  # Expected

        mock_stream.assert_called_once()
        inner_req_list = _extract_inner_req_list(mock_stream)

        self.assertEqual(
            len(inner_req_list),
            80,
            f"inner_req_list should have 80 elements with deep_think=True "
            f"but has {len(inner_req_list)}.",
        )


# ---------------------------------------------------------------------------
# R2: Public API parameter threading
#
# These tests verify that the deep_think parameter is properly accepted and
# forwarded by every public API method down to _generate(), which is where
# the flag injection into inner_req_list[49] actually happens.
#
# Methods under test:
#   - GeminiClient.generate_content()
#   - GeminiClient.generate_content_stream()
#   - ChatSession.send_message()
#   - ChatSession.send_message_stream()
#
# Note on expected behavior: Because these methods use **kwargs and forward
# them to _generate(), the parameter MAY already thread through. If tests
# pass immediately, they still serve as regression tests to ensure this
# behavior is preserved if signatures are ever refactored.
# ---------------------------------------------------------------------------


def _make_public_api_client():
    """Create a GeminiClient wired up enough for generate_content() and
    generate_content_stream() to work.

    Extends _make_generate_client with additional attributes and mocks needed
    by the public API layer (which calls _batch_execute before _generate).
    """
    client = _make_generate_client()

    # generate_content / generate_content_stream check auto_close (already False)
    # and call _batch_execute for bard_activity -- mock it out
    client._batch_execute = AsyncMock(return_value=MagicMock())

    # proxy is referenced in upload_file (only if files are provided, but
    # set it for safety)
    client.proxy = None

    # The stream mock captures the actual HTTP call inside _generate
    mock_stream = _make_stream_mock()
    client.client = MagicMock()
    client.client.stream = mock_stream
    client.client.cookies = Cookies()

    return client, mock_stream


# ---------------------------------------------------------------------------
# Test 4: generate_content forwards deep_think to _generate
# ---------------------------------------------------------------------------

class TestGenerateContentForwardsDeepThink(unittest.IsolatedAsyncioTestCase):
    """
    GeminiClient.generate_content(prompt, deep_think=True) must forward
    the deep_think parameter to _generate(), resulting in
    inner_req_list[49] == 20 in the captured StreamGenerate request body.

    This is the primary non-streaming public API method. If deep_think
    doesn't reach _generate through this path, users have no way to
    enable deep think mode via the documented API surface.
    """

    async def test_generate_content_forwards_deep_think(self):
        """
        Given deep_think=True passed to generate_content(), the underlying
        StreamGenerate request must contain inner_req_list[49] == 20.

        Why this matters: generate_content is the main public entry point.
        If the parameter doesn't thread through, deep think mode is
        inaccessible to callers.
        """
        from gemini_webapi.constants import (
            DEEP_THINK_FLAG_INDEX,
            DEEP_THINK_FLAG_VALUE,
            Model,
        )

        client, mock_stream = _make_public_api_client()

        with patch.object(
            client, "_get_waa_token", new_callable=AsyncMock, return_value=None
        ):
            try:
                await client.generate_content(
                    prompt="Explain quantum entanglement in depth",
                    model=Model.UNSPECIFIED,
                    deep_think=True,
                )
            except _StreamAbort:
                pass  # Expected -- our mock aborts the stream

        mock_stream.assert_called_once()
        inner_req_list = _extract_inner_req_list(mock_stream)

        self.assertEqual(
            inner_req_list[DEEP_THINK_FLAG_INDEX],
            DEEP_THINK_FLAG_VALUE,
            f"inner_req_list[{DEEP_THINK_FLAG_INDEX}] should be "
            f"{DEEP_THINK_FLAG_VALUE} when deep_think=True is passed to "
            f"generate_content(), but got "
            f"{inner_req_list[DEEP_THINK_FLAG_INDEX]!r}. "
            "The deep_think parameter must be forwarded from "
            "generate_content() to _generate().",
        )

    async def test_generate_content_deep_think_default_is_off(self):
        """
        When generate_content() is called without deep_think, position [49]
        must remain None -- existing callers must not be affected.

        Why this matters: Backward compatibility. If deep_think defaults
        to True or leaks through accidentally, all existing users would
        get extended-reasoning responses unexpectedly.
        """
        from gemini_webapi.constants import DEEP_THINK_FLAG_INDEX, Model

        client, mock_stream = _make_public_api_client()

        with patch.object(
            client, "_get_waa_token", new_callable=AsyncMock, return_value=None
        ):
            try:
                await client.generate_content(
                    prompt="Hello world",
                    model=Model.UNSPECIFIED,
                )
            except _StreamAbort:
                pass

        mock_stream.assert_called_once()
        inner_req_list = _extract_inner_req_list(mock_stream)

        self.assertIsNone(
            inner_req_list[DEEP_THINK_FLAG_INDEX],
            f"inner_req_list[{DEEP_THINK_FLAG_INDEX}] should be None when "
            f"deep_think is not passed to generate_content(), but got "
            f"{inner_req_list[DEEP_THINK_FLAG_INDEX]!r}.",
        )


# ---------------------------------------------------------------------------
# Test 5: send_message forwards deep_think through ChatSession
# ---------------------------------------------------------------------------

class TestSendMessageForwardsDeepThink(unittest.IsolatedAsyncioTestCase):
    """
    ChatSession.send_message(prompt, deep_think=True) must forward the
    parameter through generate_content() and ultimately to _generate().

    This exercises the two-hop forwarding path:
    send_message(**kwargs) -> generate_content(**kwargs) -> _generate(**kwargs)

    If any hop in this chain drops or misroutes the parameter, deep think
    mode becomes inaccessible from ChatSession.
    """

    async def test_send_message_forwards_deep_think(self):
        """
        Given a ChatSession created via start_chat(), calling
        send_message(prompt, deep_think=True) must result in
        inner_req_list[49] == 20 in the captured request.

        Why this matters: ChatSession is the conversation-stateful API.
        Many users prefer this interface for multi-turn interactions.
        Deep think must work here too.
        """
        from gemini_webapi.constants import (
            DEEP_THINK_FLAG_INDEX,
            DEEP_THINK_FLAG_VALUE,
            Model,
        )

        client, mock_stream = _make_public_api_client()
        chat = client.start_chat(model=Model.UNSPECIFIED)

        with patch.object(
            client, "_get_waa_token", new_callable=AsyncMock, return_value=None
        ):
            try:
                await chat.send_message(
                    prompt="Think deeply about the nature of consciousness",
                    deep_think=True,
                )
            except _StreamAbort:
                pass  # Expected

        mock_stream.assert_called_once()
        inner_req_list = _extract_inner_req_list(mock_stream)

        self.assertEqual(
            inner_req_list[DEEP_THINK_FLAG_INDEX],
            DEEP_THINK_FLAG_VALUE,
            f"inner_req_list[{DEEP_THINK_FLAG_INDEX}] should be "
            f"{DEEP_THINK_FLAG_VALUE} when deep_think=True is passed to "
            f"send_message(), but got "
            f"{inner_req_list[DEEP_THINK_FLAG_INDEX]!r}. "
            "The deep_think parameter must be forwarded through "
            "send_message() -> generate_content() -> _generate().",
        )

    async def test_send_message_deep_think_default_is_off(self):
        """
        When send_message() is called without deep_think, position [49]
        must remain None.

        Why this matters: ChatSession users who don't request deep think
        must get standard responses.
        """
        from gemini_webapi.constants import DEEP_THINK_FLAG_INDEX, Model

        client, mock_stream = _make_public_api_client()
        chat = client.start_chat(model=Model.UNSPECIFIED)

        with patch.object(
            client, "_get_waa_token", new_callable=AsyncMock, return_value=None
        ):
            try:
                await chat.send_message(prompt="Hi there")
            except _StreamAbort:
                pass

        mock_stream.assert_called_once()
        inner_req_list = _extract_inner_req_list(mock_stream)

        self.assertIsNone(
            inner_req_list[DEEP_THINK_FLAG_INDEX],
            f"inner_req_list[{DEEP_THINK_FLAG_INDEX}] should be None when "
            f"deep_think is not passed to send_message(), but got "
            f"{inner_req_list[DEEP_THINK_FLAG_INDEX]!r}.",
        )


# ---------------------------------------------------------------------------
# Test 6: generate_content_stream forwards deep_think to _generate
# ---------------------------------------------------------------------------

class TestGenerateContentStreamForwardsDeepThink(unittest.IsolatedAsyncioTestCase):
    """
    GeminiClient.generate_content_stream(prompt, deep_think=True) must
    forward the deep_think parameter to _generate(), resulting in
    inner_req_list[49] == 20.

    This is the streaming counterpart to generate_content(). Both must
    support deep_think identically.
    """

    async def test_generate_content_stream_forwards_deep_think(self):
        """
        Given deep_think=True passed to generate_content_stream(), the
        underlying StreamGenerate request must contain
        inner_req_list[49] == 20.

        Why this matters: Streaming is commonly used for long responses.
        Deep think mode produces extended reasoning output that benefits
        most from streaming -- if deep_think doesn't work in streaming
        mode, the most useful combination is broken.
        """
        from gemini_webapi.constants import (
            DEEP_THINK_FLAG_INDEX,
            DEEP_THINK_FLAG_VALUE,
            Model,
        )

        client, mock_stream = _make_public_api_client()

        with patch.object(
            client, "_get_waa_token", new_callable=AsyncMock, return_value=None
        ):
            try:
                async for _ in client.generate_content_stream(
                    prompt="Derive the Euler-Lagrange equation step by step",
                    model=Model.UNSPECIFIED,
                    deep_think=True,
                ):
                    pass  # pragma: no cover
            except _StreamAbort:
                pass  # Expected

        mock_stream.assert_called_once()
        inner_req_list = _extract_inner_req_list(mock_stream)

        self.assertEqual(
            inner_req_list[DEEP_THINK_FLAG_INDEX],
            DEEP_THINK_FLAG_VALUE,
            f"inner_req_list[{DEEP_THINK_FLAG_INDEX}] should be "
            f"{DEEP_THINK_FLAG_VALUE} when deep_think=True is passed to "
            f"generate_content_stream(), but got "
            f"{inner_req_list[DEEP_THINK_FLAG_INDEX]!r}. "
            "The deep_think parameter must be forwarded from "
            "generate_content_stream() to _generate().",
        )

    async def test_generate_content_stream_deep_think_default_is_off(self):
        """
        When generate_content_stream() is called without deep_think,
        position [49] must remain None.

        Why this matters: Streaming callers who don't request deep think
        must not get altered behavior.
        """
        from gemini_webapi.constants import DEEP_THINK_FLAG_INDEX, Model

        client, mock_stream = _make_public_api_client()

        with patch.object(
            client, "_get_waa_token", new_callable=AsyncMock, return_value=None
        ):
            try:
                async for _ in client.generate_content_stream(
                    prompt="Hello",
                    model=Model.UNSPECIFIED,
                ):
                    pass  # pragma: no cover
            except _StreamAbort:
                pass

        mock_stream.assert_called_once()
        inner_req_list = _extract_inner_req_list(mock_stream)

        self.assertIsNone(
            inner_req_list[DEEP_THINK_FLAG_INDEX],
            f"inner_req_list[{DEEP_THINK_FLAG_INDEX}] should be None when "
            f"deep_think is not passed to generate_content_stream(), but got "
            f"{inner_req_list[DEEP_THINK_FLAG_INDEX]!r}.",
        )


# ---------------------------------------------------------------------------
# Test 7: send_message_stream forwards deep_think through ChatSession
# ---------------------------------------------------------------------------

class TestSendMessageStreamForwardsDeepThink(unittest.IsolatedAsyncioTestCase):
    """
    ChatSession.send_message_stream(prompt, deep_think=True) must forward
    the parameter through generate_content_stream() and to _generate().

    This exercises the streaming two-hop forwarding path:
    send_message_stream(**kwargs) -> generate_content_stream(**kwargs) -> _generate(**kwargs)
    """

    async def test_send_message_stream_forwards_deep_think(self):
        """
        Given a ChatSession, calling send_message_stream(prompt, deep_think=True)
        must result in inner_req_list[49] == 20 in the captured request.

        Why this matters: This is the streaming + conversation-stateful
        combination. Users doing multi-turn deep reasoning need both
        session state and streaming to work with deep_think.
        """
        from gemini_webapi.constants import (
            DEEP_THINK_FLAG_INDEX,
            DEEP_THINK_FLAG_VALUE,
            Model,
        )

        client, mock_stream = _make_public_api_client()
        chat = client.start_chat(model=Model.UNSPECIFIED)

        with patch.object(
            client, "_get_waa_token", new_callable=AsyncMock, return_value=None
        ):
            try:
                async for _ in chat.send_message_stream(
                    prompt="Analyze the implications of Godel's incompleteness theorems",
                    deep_think=True,
                ):
                    pass  # pragma: no cover
            except _StreamAbort:
                pass  # Expected

        mock_stream.assert_called_once()
        inner_req_list = _extract_inner_req_list(mock_stream)

        self.assertEqual(
            inner_req_list[DEEP_THINK_FLAG_INDEX],
            DEEP_THINK_FLAG_VALUE,
            f"inner_req_list[{DEEP_THINK_FLAG_INDEX}] should be "
            f"{DEEP_THINK_FLAG_VALUE} when deep_think=True is passed to "
            f"send_message_stream(), but got "
            f"{inner_req_list[DEEP_THINK_FLAG_INDEX]!r}. "
            "The deep_think parameter must be forwarded through "
            "send_message_stream() -> generate_content_stream() -> _generate().",
        )


if __name__ == "__main__":
    unittest.main()
