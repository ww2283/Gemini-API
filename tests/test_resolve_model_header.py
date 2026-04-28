"""Tests for ``GeminiClient._resolve_model_header`` -- per-request slot-16 UUID.

The jspb header ``x-goog-ext-525001261-jspb`` carries a UUID at slot 16 that
Google validates as a per-request nonce. Today the harvester captures it once
into ``~/.cache/gemini_webapi/jspb_templates.json`` (24h TTL) and the runtime
replays the same UUID across every request via ``model.model_header``. After a
2026-04-28 server-side tightening, replayed slot-16 nonces are rejected with
batch-execute status [5].

The fix mirrors the precedent set by ``inner_req_list[59]`` (see
``client.py:1074``): inject a fresh ``str(uuid.uuid4()).upper()`` into the
returned header on every call to ``_resolve_model_header``. These tests pin
that contract.
"""
from __future__ import annotations

import json as _json
import re

import pytest

from gemini_webapi.client import GeminiClient
from gemini_webapi.constants import Model

_JSPB = "x-goog-ext-525001261-jspb"
_UUID_RE = re.compile(
    r"^[0-9A-F]{8}-[0-9A-F]{4}-[0-9A-F]{4}-[0-9A-F]{4}-[0-9A-F]{12}$"
)


@pytest.fixture(autouse=True)
def restore_model_headers():
    """Snapshot and restore model headers around each test.

    Tests in this module mutate ``Model.G_3_1_PRO.model_header`` directly to
    set up the input shape for ``_resolve_model_header``. Without this
    fixture, mutations bleed across tests and into the rest of the suite.
    """
    snapshot = {m: dict(m.model_header) for m in Model if m is not Model.UNSPECIFIED}
    try:
        yield
    finally:
        for m, header in snapshot.items():
            m.model_header.clear()
            m.model_header.update(header)


def _make_client() -> GeminiClient:
    """Construct a minimal ``GeminiClient`` without running ``__init__``.

    Skips the async network bring-up and only sets the attributes
    ``_resolve_model_header`` reads. ``_discovered_model_ids`` is empty so
    the model-ID resolution branch is a no-op -- isolating slot-16 UUID
    injection as the only behavior under test. ``_MODEL_TYPE_MAP`` is a
    class attribute on ``GeminiClient`` so we don't need to set it.
    """
    client = GeminiClient.__new__(GeminiClient)
    client._discovered_model_ids = {}
    return client


def _set_pro_jspb(jspb_list: list) -> None:
    """Replace Pro's jspb header with a known shape for the test."""
    Model.G_3_1_PRO.model_header[_JSPB] = _json.dumps(jspb_list)


def _slot16_of(header: dict) -> object:
    """Parse the returned header's jspb and return slot 16 (or KeyError)."""
    raw = header[_JSPB]
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    parsed = _json.loads(raw)
    return parsed[16]


# ---------------------------------------------------------------------------
# R1 contract: per-request fresh UUID at jspb slot 16
# ---------------------------------------------------------------------------


def test_resolve_returns_fresh_uuid_at_slot_16():
    """Slot 16 in the returned header must NOT equal the cached value.

    The cached UUID is what the harvester wrote into ``model.model_header``
    after one capture. If the runtime replays it, Google rejects the
    request with status [5]. The fix injects a fresh UUID per call, so the
    returned slot 16 must differ from the input.
    """
    cached_uuid = "EB924281-8BD2-4EFF-96E0-7DB22E1CC347"
    _set_pro_jspb(
        [
            1, None, None, None, None, None, None, 0, [4],
            None, None, 3, None, None, 3, None, cached_uuid,
        ]
    )
    client = _make_client()

    header = client._resolve_model_header(Model.G_3_1_PRO)
    slot16 = _slot16_of(header)

    assert isinstance(slot16, str), (
        f"slot 16 must be a string UUID, got {type(slot16).__name__}: {slot16!r}"
    )
    assert slot16, "slot 16 must be a non-empty string"
    assert slot16 != cached_uuid, (
        f"slot 16 was replayed verbatim from the cached value ({cached_uuid!r}); "
        f"runtime must inject a fresh UUID per request"
    )


def test_resolve_returns_different_uuid_on_each_call():
    """Two consecutive calls must produce two different slot-16 values.

    Pins per-call freshness rather than memoization. If the implementation
    were to cache the generated UUID on the client or model, this test
    would catch it.
    """
    _set_pro_jspb(
        [
            1, None, None, None, None, None, None, 0, [4],
            None, None, 3, None, None, 3, None, None,
        ]
    )
    client = _make_client()

    first = _slot16_of(client._resolve_model_header(Model.G_3_1_PRO))
    second = _slot16_of(client._resolve_model_header(Model.G_3_1_PRO))

    assert first != second, (
        f"slot 16 was identical across two calls ({first!r}); UUID must be "
        f"freshly generated per request, not memoized"
    )


def test_resolve_preserves_other_slots():
    """Only slot 16 changes; slots 0..15 must round-trip unchanged.

    Guards against an over-broad implementation that rebuilds the entire
    template instead of surgically replacing slot 16. Also asserts slot 16
    *did* change -- otherwise, in the broken-today state where no fresh
    UUID is injected, the "all slots unchanged" check would trivially pass.
    """
    input_jspb = [
        1, None, None, None, None, None, None, 0, [4],
        None, None, 3, None, None, 3, None, "OLD-UUID",
    ]
    _set_pro_jspb(input_jspb)
    client = _make_client()

    header = client._resolve_model_header(Model.G_3_1_PRO)
    raw = header[_JSPB]
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    parsed = _json.loads(raw)

    assert len(parsed) >= 17, (
        f"returned jspb must keep at least 17 slots, got {len(parsed)}"
    )
    for i in range(16):
        assert parsed[i] == input_jspb[i], (
            f"slot {i} was mutated: input={input_jspb[i]!r} "
            f"-> output={parsed[i]!r}; only slot 16 should change"
        )
    assert parsed[16] != "OLD-UUID", (
        f"slot 16 was not refreshed: still {parsed[16]!r}. The fix must "
        f"inject a fresh UUID per request, not replay the cached value"
    )


def test_resolve_uppercase_uuid_format():
    """Slot 16 must match Chrome's wire format: uppercase v4 UUID.

    Regex: ``^[0-9A-F]{8}-[0-9A-F]{4}-[0-9A-F]{4}-[0-9A-F]{4}-[0-9A-F]{12}$``.
    Using ``re.fullmatch`` so any stray whitespace or extra chars fail.
    Mirrors how live Chrome captures format the slot.
    """
    _set_pro_jspb(
        [
            1, None, None, None, None, None, None, 0, [4],
            None, None, 3, None, None, 3, None, None,
        ]
    )
    client = _make_client()

    header = client._resolve_model_header(Model.G_3_1_PRO)
    slot16 = _slot16_of(header)

    assert isinstance(slot16, str), (
        f"slot 16 must be a string, got {type(slot16).__name__}: {slot16!r}"
    )
    assert re.fullmatch(_UUID_RE, slot16), (
        f"slot 16 {slot16!r} is not an uppercase v4 UUID matching "
        f"{_UUID_RE.pattern}"
    )
