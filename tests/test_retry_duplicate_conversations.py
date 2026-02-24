"""
Tests for the retry-abort guard that prevents duplicate conversations when
a ChatSession's cid gets assigned mid-stream and then an APIError triggers
a retry via the @running decorator.

These tests define the expected behavior for the fix described in the bug report:
when _generate() is retried after an APIError, and the ChatSession started with
an empty cid but now has a truthy cid (assigned by Gemini mid-stream), the retry
should be aborted by raising GeminiError (which the @running decorator does NOT
catch) instead of allowing a duplicate conversation to be created.

All tests are expected to FAIL until the guard logic is implemented in
GeminiClient._generate().
"""

import asyncio
import time
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from gemini_webapi.client import GeminiClient, ChatSession
from gemini_webapi.exceptions import APIError, GeminiError


def _make_running_client() -> GeminiClient:
    """Create a minimal GeminiClient that appears initialized (running).

    The client is configured so the @running decorator sees _running=True
    and does not attempt real initialization. The close() method is mocked
    as a no-op to prevent _running from being flipped to False during the
    status-500 error path in _generate().
    """
    client = GeminiClient.__new__(GeminiClient)
    # Attributes required by the @running decorator
    client._running = True
    client.timeout = 30
    client.auto_close = False
    client.close_delay = 300
    client.auto_refresh = False
    client.refresh_interval = 540
    client.verbose = False
    client.watchdog_timeout = 60
    # Attributes required by _generate() internals
    client._reqid = 10000
    client.access_token = "fake_token"
    client.build_label = None
    client.session_id = None
    client.cookies = MagicMock()
    client.client = AsyncMock()
    client.proxy = None
    client.kwargs = {}
    client._lock = asyncio.Lock()
    client.close_task = None
    client.refresh_task = None
    # Mock close() as a no-op so _running stays True across retries.
    # In production, close() sets _running=False, which would cause the
    # decorator to call init() -- requiring real auth cookies.
    client.close = AsyncMock()
    return client


def _make_chat_session(client: GeminiClient, cid: str = "") -> ChatSession:
    """Create a ChatSession with the given initial cid."""
    chat = ChatSession.__new__(ChatSession)
    # Manually initialize the private metadata list (matches ChatSession.__init__)
    chat._ChatSession__metadata = [
        cid, "", "", None, None, None, None, None, None, ""
    ]
    chat.geminiclient = client
    chat.last_output = None
    chat.model = "unspecified"
    chat.gem = None
    return chat


def _make_fail_stream(status_code=500, on_enter=None):
    """Return a callable that produces async context managers simulating a
    failed HTTP stream response.

    Parameters
    ----------
    status_code : int
        The HTTP status code the mock response will report.
    on_enter : callable or None
        Optional side-effect function called each time the stream context
        is entered (useful for mutating chat.cid mid-stream).
    """
    mock_response = AsyncMock()
    mock_response.status_code = status_code

    class _FailStreamCtx:
        async def __aenter__(self_ctx):
            if on_enter is not None:
                on_enter()
            return mock_response

        async def __aexit__(self_ctx, *args):
            pass

    def stream_factory(*args, **kwargs):
        return _FailStreamCtx()

    return stream_factory


class TestRetryAbortOnCidAssignedMidstream(unittest.IsolatedAsyncioTestCase):
    """
    When a ChatSession starts with an empty cid, Gemini may return metadata
    that assigns a real cid during the first attempt of _generate(). If an
    APIError then occurs and the @running decorator retries, the guard in
    _generate() must detect the cid was empty at the start but is now truthy,
    and raise GeminiError to abort the retry -- preventing a duplicate
    conversation.
    """

    async def test_retry_aborts_when_cid_assigned_midstream(self):
        """
        Scenario:
          1. ChatSession starts with cid="" (new conversation).
          2. During the first call to _generate(), the HTTP response returns
             status 500. Before the APIError is raised, the close() side-effect
             simulates Gemini having assigned a cid (as would happen at line 692
             if some metadata arrived before the stream broke).
          3. The @running decorator catches APIError and retries _generate().
          4. On retry entry, the guard should detect that original_cid was ""
             but chat.cid is now truthy, and raise GeminiError to abort.

        Expected: GeminiError is raised (not APIError), aborting the retry
        and preventing a duplicate conversation on Gemini's web UI.
        """
        client = _make_running_client()
        chat = _make_chat_session(client, cid="")

        # Verify precondition
        self.assertEqual(chat.cid, "")

        session_state = {
            "last_texts": {},
            "last_thoughts": {},
            "last_progress_time": time.time(),
        }

        stream_call_count = 0

        def on_stream_enter():
            """Side-effect: on the first stream entry, simulate Gemini having
            assigned a cid via metadata before the stream broke."""
            nonlocal stream_call_count
            stream_call_count += 1
            if stream_call_count == 1:
                # Simulates what happens at line 692 when Gemini returns
                # metadata with a conversation id before the stream fails.
                chat.cid = "c_abc123_assigned_midstream"

        client.client.stream = _make_fail_stream(
            status_code=500, on_enter=on_stream_enter
        )

        # The flow:
        # 1. _generate() enters, should save original_cid="" in session_state
        # 2. Stream returns 500 -> calls self.close() (mocked no-op) -> APIError
        # 3. @running catches APIError, retries with same session_state + kwargs
        # 4. _generate() re-enters. Guard checks: original_cid=="" but
        #    chat.cid="c_abc123_assigned_midstream" -> raises GeminiError
        # 5. GeminiError is NOT caught by @running -> propagates to caller
        with patch("gemini_webapi.utils.decorators.DELAY_FACTOR", 0):
            with self.assertRaises(GeminiError) as ctx:
                async for _ in client._generate(
                    prompt="test prompt",
                    chat=chat,
                    session_state=session_state,
                ):
                    pass

        self.assertIn(
            "duplicate",
            str(ctx.exception).lower(),
            "GeminiError message should mention duplicate conversation prevention",
        )
        # The guard should have fired on the 2nd call, meaning stream was
        # only entered once (the first call).
        self.assertEqual(stream_call_count, 1,
            "Stream should only be entered once; the retry should be aborted "
            "before reaching the HTTP call.")


class TestRetryProceedsWhenCidAlreadySet(unittest.IsolatedAsyncioTestCase):
    """
    When a ChatSession already has a cid before _generate() is called
    (continuing an existing conversation), an APIError should NOT abort the
    retry -- the cid was already set, so retrying won't create a duplicate.
    """

    async def test_retry_proceeds_when_cid_was_already_set(self):
        """
        Scenario:
          1. ChatSession starts with cid="c_existing_123" (existing conversation).
          2. Every call to _generate() hits status 500 -> APIError.
          3. The @running decorator retries until retries are exhausted.
          4. The guard should NOT intervene because original_cid was truthy.

        Expected: APIError is raised after all retries are exhausted
        (not GeminiError), proving the guard allowed normal retry behavior.
        """
        client = _make_running_client()
        chat = _make_chat_session(client, cid="c_existing_123")

        session_state = {
            "last_texts": {},
            "last_thoughts": {},
            "last_progress_time": time.time(),
        }

        stream_call_count = 0

        def count_stream_entries():
            nonlocal stream_call_count
            stream_call_count += 1

        client.client.stream = _make_fail_stream(
            status_code=500, on_enter=count_stream_entries
        )

        with patch("gemini_webapi.utils.decorators.DELAY_FACTOR", 0):
            with self.assertRaises(APIError):
                async for _ in client._generate(
                    prompt="test prompt",
                    chat=chat,
                    session_state=session_state,
                ):
                    pass

        # With retry=5 and the original call, we expect 6 total stream entries
        # (1 original + 5 retries). All should proceed because cid was already set.
        self.assertGreater(stream_call_count, 1,
            "Multiple stream attempts should occur when cid was already set, "
            "proving the guard did not abort retries.")


class TestRetryProceedsWhenNoChatSession(unittest.IsolatedAsyncioTestCase):
    """
    When _generate() is called without a ChatSession (chat=None), the guard
    logic should not interfere -- there is no ChatSession to track, so retries
    should proceed normally.
    """

    async def test_retry_proceeds_when_no_chat_session(self):
        """
        Scenario:
          1. _generate() is called with chat=None (standalone generation).
          2. Every call hits status 500 -> APIError.
          3. The @running decorator retries until exhausted.
          4. The guard should NOT intervene (no ChatSession).

        Expected: APIError is raised after all retries (not GeminiError).
        """
        client = _make_running_client()

        session_state = {
            "last_texts": {},
            "last_thoughts": {},
            "last_progress_time": time.time(),
        }

        stream_call_count = 0

        def count_stream_entries():
            nonlocal stream_call_count
            stream_call_count += 1

        client.client.stream = _make_fail_stream(
            status_code=500, on_enter=count_stream_entries
        )

        with patch("gemini_webapi.utils.decorators.DELAY_FACTOR", 0):
            with self.assertRaises(APIError):
                async for _ in client._generate(
                    prompt="test prompt",
                    chat=None,
                    session_state=session_state,
                ):
                    pass

        # Should have multiple stream attempts (no guard to abort).
        self.assertGreater(stream_call_count, 1,
            "Multiple stream attempts should occur when chat=None, "
            "proving the guard did not interfere.")


class TestSessionStateTracksOriginalCid(unittest.IsolatedAsyncioTestCase):
    """
    Verify that session_state["original_cid"] is set on first entry to
    _generate() and preserved across retries. This is the foundational
    mechanism for the duplicate-conversation guard.
    """

    async def test_session_state_tracks_original_cid(self):
        """
        Scenario:
          1. A ChatSession starts with cid="" (new conversation).
          2. session_state is passed to _generate().
          3. On first entry, _generate() should set
             session_state["original_cid"] = "" (the current chat.cid).
          4. Even after _generate() raises and retries, session_state should
             still contain "original_cid" == "".

        Expected: After _generate() completes (even with failure),
        session_state contains "original_cid" set to the chat.cid value
        from the very first entry.
        """
        client = _make_running_client()
        chat = _make_chat_session(client, cid="")

        session_state = {
            "last_texts": {},
            "last_thoughts": {},
            "last_progress_time": time.time(),
        }

        # Verify precondition: no original_cid key yet
        self.assertNotIn("original_cid", session_state)

        client.client.stream = _make_fail_stream(status_code=500)

        with patch("gemini_webapi.utils.decorators.DELAY_FACTOR", 0):
            try:
                async for _ in client._generate(
                    prompt="test prompt",
                    chat=chat,
                    session_state=session_state,
                ):
                    pass
            except (APIError, GeminiError):
                pass  # Expected -- we just want to inspect session_state

        # The critical assertion: session_state should now contain original_cid.
        # This will FAIL because _generate() does not yet track original_cid.
        self.assertIn(
            "original_cid",
            session_state,
            "session_state must track 'original_cid' to detect cid changes "
            "across retries. This key should be set on first entry to _generate().",
        )
        self.assertEqual(
            session_state["original_cid"],
            "",
            "original_cid should be the empty string that chat.cid had when "
            "_generate() was first entered.",
        )


if __name__ == "__main__":
    unittest.main()
