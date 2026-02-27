"""
Tests for GeminiClient.read_chat(cid) -- Phase A+B of response recovery.

read_chat() fetches a conversation's content by cid using the GRPC.READ_CHAT
RPC, parses the batchexecute response, extracts the latest assistant message,
and returns a ModelOutput (or None on failure).
"""

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock

import orjson as json

from gemini_webapi.client import GeminiClient
from gemini_webapi.constants import GRPC
from gemini_webapi.types import ModelOutput, Candidate, RPCData


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_running_client() -> GeminiClient:
    """Create a minimal GeminiClient that appears initialized (running).

    Reuses the pattern from test_retry_duplicate_conversations.py.
    """
    client = GeminiClient.__new__(GeminiClient)
    client._running = True
    client.timeout = 30
    client.auto_close = False
    client.close_delay = 300
    client.auto_refresh = False
    client.refresh_interval = 540
    client.verbose = False
    client.watchdog_timeout = 60
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
    client.close = AsyncMock()
    return client


def _build_batchexecute_response(inner_json: list) -> str:
    """Build a realistic batchexecute response envelope.

    Google's batchexecute wraps each RPC response in a JSON array where:
      - part[0] is the RPC method identifier string (e.g. "wrb.fr")
      - part[1] is the RPC id (e.g. "hNvQHb")
      - part[2] is a JSON-encoded string containing the actual data
      - remaining elements are metadata

    The outer response starts with )]}\' anti-XSSI prefix and uses a
    length-prefixed framing protocol.
    """
    # Build the inner part: [rpc_wrapper, rpc_id, json_encoded_body, ...]
    inner_encoded = json.dumps(inner_json).decode("utf-8")
    part = ["wrb.fr", "hNvQHb", inner_encoded, None, None, None, "generic"]
    # Wrap in the outer envelope as a single-element list-of-lists
    envelope = json.dumps([part]).decode("utf-8")
    # Add the anti-XSSI prefix and length-prefixed frame
    frame_len = len(envelope.encode("utf-16-le")) // 2  # UTF-16 code units
    return f")]}}'\n\n{frame_len}\n{envelope}\n"


def _make_mock_response(text: str) -> MagicMock:
    """Create a mock httpx.Response with the given text."""
    response = MagicMock()
    response.text = text
    return response


# ---------------------------------------------------------------------------
# Verified READ_CHAT response structure (from live API testing 2026-02-24).
#
# After json.loads(part[2]), the inner JSON is:
#   part_body[0][0] = conversation turn object:
#     [0] = [cid, rid]            # metadata
#     [1] = null
#     [2] = [[user_text, ...]]    # user's message content
#     [3] = assistant response object:
#       [0] = [candidate, ...]    # candidates list
#         candidate[0] = rcid     # reply candidate ID (e.g. "rc_...")
#         candidate[1] = [text]   # text content at [1][0]
#         candidate[37] = thoughts (optional, JSON string)
#       [3] = rcid (duplicate)
#     [4] = [timestamp, nanos]
# ---------------------------------------------------------------------------

SAMPLE_CID = "c_abc123def456"
SAMPLE_RID = "r_f0cfdd6e03bfeea2"
SAMPLE_RCID = "rc_reply_001"


def _make_candidate(rcid, text):
    """Build a candidate array matching the real Gemini structure.

    In production, candidates have 38 elements. We only populate the
    indices that read_chat() accesses: [0]=rcid, [1]=[text].
    """
    cand = [None] * 38
    cand[0] = rcid
    cand[1] = [text]
    return cand


# A single-turn conversation (one user message + one assistant response)
SINGLE_TURN_INNER = [
    [  # part_body[0] — list of conversation turns
        [  # part_body[0][0] — single conversation turn
            [SAMPLE_CID, SAMPLE_RID],              # [0]: metadata
            None,                                    # [1]: null
            [["What is 2+2?"]],                      # [2]: user message
            [                                        # [3]: assistant response
                [_make_candidate(SAMPLE_RCID, "The answer is 4.")],  # [0]: candidates
            ],
            [1771949743, 7000000],                   # [4]: timestamp
        ],
    ],
]

# For multi-candidate test: same structure but with a different rcid/text
SAMPLE_CONVERSATION_INNER = [
    [
        [
            [SAMPLE_CID, SAMPLE_RID],
            None,
            [["Tell me about Python."]],
            [
                [_make_candidate(SAMPLE_RCID, "Python is a versatile programming language.")],
            ],
            [1771949743, 7000000],
        ],
    ],
]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestReadChatCallsBatchExecute(unittest.IsolatedAsyncioTestCase):
    """Verify read_chat calls _batch_execute with the correct RPC and payload."""

    async def test_read_chat_calls_batch_execute_with_correct_rpc(self):
        """
        read_chat(cid) must call _batch_execute with:
          - GRPC.READ_CHAT as the rpcid
          - A payload containing the cid and the correct structure:
            json.dumps([cid, 10, None, 1, [0], [4], None, 1])

        This test mocks _batch_execute to capture its arguments and verifies
        the RPC is constructed correctly, without caring about response parsing.
        """
        client = _make_running_client()
        cid = "c_test_conversation_123"

        # Mock _batch_execute to return a minimal valid-looking response
        mock_response = _make_mock_response("")
        client._batch_execute = AsyncMock(return_value=mock_response)

        # Act -- call the method under test
        await client.read_chat(cid)

        # Assert -- _batch_execute was called exactly once
        client._batch_execute.assert_called_once()

        # Extract the RPCData from the call
        call_args = client._batch_execute.call_args
        payloads = call_args[0][0]  # first positional arg is list[RPCData]
        self.assertEqual(len(payloads), 1, "Should send exactly one RPC in the batch")

        rpc_data = payloads[0]
        self.assertIsInstance(rpc_data, RPCData)
        self.assertEqual(rpc_data.rpcid, GRPC.READ_CHAT)

        # Verify the payload contains the cid in the expected structure
        payload_parsed = json.loads(rpc_data.payload)
        self.assertEqual(
            payload_parsed[0], cid,
            "First element of the READ_CHAT payload must be the conversation id"
        )
        # Verify the full payload structure
        expected_payload = [cid, 10, None, 1, [0], [4], None, 1]
        self.assertEqual(payload_parsed, expected_payload)


class TestReadChatReturnsModelOutput(unittest.IsolatedAsyncioTestCase):
    """Verify read_chat returns a properly constructed ModelOutput."""

    async def test_read_chat_returns_model_output_from_valid_response(self):
        """
        Given a realistic batchexecute response containing a conversation with
        a single assistant reply, read_chat should return a ModelOutput with:
          - metadata containing the cid
          - a Candidate with the correct text and rcid
        """
        client = _make_running_client()

        response_text = _build_batchexecute_response(SINGLE_TURN_INNER)
        mock_response = _make_mock_response(response_text)
        client._batch_execute = AsyncMock(return_value=mock_response)

        # Act
        result = await client.read_chat(SAMPLE_CID)

        # Assert -- should return a ModelOutput, not None
        self.assertIsNotNone(result, "read_chat should return a ModelOutput for valid responses")
        self.assertIsInstance(result, ModelOutput)

        # Verify the text content was extracted from the assistant message
        self.assertEqual(
            result.text, "The answer is 4.",
            "ModelOutput.text should contain the assistant's reply text"
        )

        # Verify metadata includes the cid
        self.assertIn(
            SAMPLE_CID, result.metadata,
            "ModelOutput metadata should include the conversation id"
        )

        # Verify the candidate has the correct rcid
        self.assertEqual(len(result.candidates), 1)
        self.assertEqual(result.candidates[0].rcid, SAMPLE_RCID)


class TestReadChatReturnsNoneOnEmptyResponse(unittest.IsolatedAsyncioTestCase):
    """Verify graceful degradation when response has no parseable content."""

    async def test_read_chat_returns_none_on_empty_response(self):
        """
        When _batch_execute returns a response with no meaningful content
        (empty body, no conversation data), read_chat should return None
        rather than raising an exception. This supports graceful degradation
        in the retry-recovery flow.
        """
        client = _make_running_client()

        # An empty batchexecute response -- the inner part body is an empty list
        response_text = _build_batchexecute_response([])
        mock_response = _make_mock_response(response_text)
        client._batch_execute = AsyncMock(return_value=mock_response)

        # Act
        result = await client.read_chat(SAMPLE_CID)

        # Assert -- should return None, not raise
        self.assertIsNone(
            result,
            "read_chat should return None when the response contains no conversation data"
        )


class TestReadChatReturnsNoneOnParseError(unittest.IsolatedAsyncioTestCase):
    """Verify read_chat handles malformed responses gracefully."""

    async def test_read_chat_returns_none_on_parse_error(self):
        """
        When _batch_execute returns garbled/unparseable text, read_chat should
        catch the error and return None instead of letting the exception
        propagate. This is critical for the recovery path -- if reading the
        chat fails, we fall back to returning None rather than crashing.
        """
        client = _make_running_client()

        # Completely garbled response that will fail JSON parsing
        mock_response = _make_mock_response("this is not valid json at all {{{")
        client._batch_execute = AsyncMock(return_value=mock_response)

        # Act -- should NOT raise any exception
        result = await client.read_chat(SAMPLE_CID)

        # Assert
        self.assertIsNone(
            result,
            "read_chat should return None when response parsing fails, not raise"
        )

    async def test_read_chat_returns_none_when_batch_execute_raises(self):
        """
        If _batch_execute itself raises an exception (network error, auth
        failure, etc.), read_chat should catch it and return None.
        """
        client = _make_running_client()
        client._batch_execute = AsyncMock(side_effect=Exception("Network timeout"))

        result = await client.read_chat(SAMPLE_CID)

        self.assertIsNone(
            result,
            "read_chat should return None when _batch_execute raises an exception"
        )


class TestReadChatExtractsFromConversationStructure(unittest.IsolatedAsyncioTestCase):
    """Verify read_chat correctly navigates the nested conversation structure."""

    async def test_read_chat_extracts_from_conversation_turn(self):
        """
        The READ_CHAT response contains a conversation turn with the assistant's
        response nested at turn[3][0][candidate]. Verify read_chat navigates
        this structure and extracts the correct text and rcid.
        """
        client = _make_running_client()

        response_text = _build_batchexecute_response(SAMPLE_CONVERSATION_INNER)
        mock_response = _make_mock_response(response_text)
        client._batch_execute = AsyncMock(return_value=mock_response)

        # Act
        result = await client.read_chat(SAMPLE_CID)

        # Assert
        self.assertIsNotNone(result)
        self.assertIsInstance(result, ModelOutput)

        self.assertEqual(
            result.text,
            "Python is a versatile programming language.",
            "read_chat must extract the assistant's response text from the conversation turn"
        )
        self.assertEqual(
            result.candidates[0].rcid,
            SAMPLE_RCID,
            "The rcid should be extracted from the candidate structure"
        )


class TestReadChatReturnsRidInMetadata(unittest.IsolatedAsyncioTestCase):
    """
    Verify that read_chat() includes the rid (reply id) in the returned
    ModelOutput.metadata, not just the cid.

    The READ_CHAT response structure has conv_turn[0] = [cid, rid].
    Before the fix, read_chat() returned metadata=[cid] without the rid,
    so after recovery ChatSession.rid was stale, causing conversation forks.
    This test verifies the fix: metadata now includes [cid, rid].
    """

    async def test_read_chat_includes_rid_in_metadata(self):
        """
        Given a valid READ_CHAT response where conv_turn[0] = [cid, rid],
        read_chat() should return ModelOutput with metadata containing BOTH
        cid and rid:
          - metadata[0] == SAMPLE_CID
          - metadata[1] == SAMPLE_RID

        This is critical for continuation recovery: when ChatSession.metadata
        is updated from the recovered ModelOutput, the rid (at index 1) must
        be present so that the next turn references the correct reply.
        """
        client = _make_running_client()

        response_text = _build_batchexecute_response(SINGLE_TURN_INNER)
        mock_response = _make_mock_response(response_text)
        client._batch_execute = AsyncMock(return_value=mock_response)

        # Act
        result = await client.read_chat(SAMPLE_CID)

        # Precondition: result should exist and be a ModelOutput
        self.assertIsNotNone(result, "read_chat should return a ModelOutput")
        self.assertIsInstance(result, ModelOutput)

        # Assert that metadata has at least 2 elements (cid and rid)
        self.assertGreaterEqual(
            len(result.metadata), 2,
            f"ModelOutput.metadata should contain at least [cid, rid], "
            f"but got {result.metadata!r} (length {len(result.metadata)}). "
            f"The rid from conv_turn[0][1] is being discarded."
        )

        # Assert metadata[0] is the cid
        self.assertEqual(
            result.metadata[0], SAMPLE_CID,
            "metadata[0] should be the conversation id (cid)"
        )

        self.assertEqual(
            result.metadata[1], SAMPLE_RID,
            f"metadata[1] should be the reply id (rid) '{SAMPLE_RID}'. "
            f"If this fails, rid extraction from conv_turn[0][1] may have regressed."
        )

    async def test_read_chat_rid_propagates_to_chat_session(self):
        """
        When a ChatSession updates its metadata from a recovered ModelOutput,
        the rid should be set correctly. This is an integration-level check
        that the metadata structure from read_chat() is compatible with
        ChatSession.metadata setter.

        ChatSession.metadata setter updates __metadata[i] for each non-None
        element. With metadata=[cid, rid], both metadata[0] (cid) and
        metadata[1] (rid) should be updated.
        """
        client = _make_running_client()

        response_text = _build_batchexecute_response(SINGLE_TURN_INNER)
        mock_response = _make_mock_response(response_text)
        client._batch_execute = AsyncMock(return_value=mock_response)

        # Act
        result = await client.read_chat(SAMPLE_CID)
        self.assertIsNotNone(result)

        # Create a ChatSession and assign the recovered metadata
        from gemini_webapi.client import ChatSession
        chat = ChatSession.__new__(ChatSession)
        chat._ChatSession__metadata = [
            "c_existing_123", "r_old_rid", "", None, None, None, None, None, None, ""
        ]

        # Simulate what happens in the recovery path:
        # ChatSession.__setattr__ triggers metadata update when last_output is set
        chat.metadata = result.metadata

        # After recovery, cid should be updated to the recovered cid
        self.assertEqual(
            chat.cid, SAMPLE_CID,
            "ChatSession.cid should be updated from recovered metadata"
        )

        # Verifies rid is propagated from recovered metadata (regression guard)
        self.assertEqual(
            chat.rid, SAMPLE_RID,
            f"ChatSession.rid should be '{SAMPLE_RID}' after recovery, "
            f"but was '{chat.rid}'. rid extraction may have regressed."
        )


class TestReadChatRidFallbackWhenMetadataNotList(unittest.IsolatedAsyncioTestCase):
    """Verify read_chat() handles turn_metadata that is not a list (e.g., scalar cid)."""

    async def test_rid_omitted_from_metadata_when_turn_metadata_is_string(self):
        """When conv_turn[0] is a string instead of [cid, rid], rid should not
        be included in metadata to avoid overwriting a valid existing rid."""
        client = _make_running_client()

        # Build inner with turn_metadata as a plain string instead of [cid, rid]
        inner = [
            [
                [
                    SAMPLE_CID,  # [0]: scalar metadata (not a list)
                    None,
                    [["What is 2+2?"]],
                    [
                        [_make_candidate(SAMPLE_RCID, "The answer is 4.")],
                    ],
                    [1771949743, 7000000],
                ],
            ],
        ]

        response_text = _build_batchexecute_response(inner)
        mock_response = _make_mock_response(response_text)
        client._batch_execute = AsyncMock(return_value=mock_response)

        result = await client.read_chat(SAMPLE_CID)

        self.assertIsNotNone(result)
        # metadata should only contain cid (no rid since extraction failed)
        self.assertEqual(result.metadata, [SAMPLE_CID])


if __name__ == "__main__":
    unittest.main()
