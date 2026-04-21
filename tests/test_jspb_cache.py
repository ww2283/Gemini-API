"""Tests for gemini_webapi.utils.jspb_cache — per-model jspb template cache.

R1 Red phase: pins the public contract of the not-yet-implemented
``jspb_cache`` module. These tests must fail at import time until G1
creates ``src/gemini_webapi/utils/jspb_cache.py``.

Contract under test (from .doc/plans/2026-04-21_per_model_template_capture.md):

- ``read_cache``     : returns dict if fresh + schema-valid, else None.
- ``write_cache``    : atomic write with captured_at + source recorded.
- ``invalidate_cache``: removes cache unless debounced by recent write.
- ``is_cache_fresh`` : True iff read_cache would return non-None.

Time and path are injected explicitly so no real clock or $HOME is touched.
"""
from __future__ import annotations

import json

import pytest

from gemini_webapi.utils.jspb_cache import (
    invalidate_cache,
    is_cache_fresh,
    read_cache,
    write_cache,
)


HOUR = 3600.0
DAY = 24 * HOUR


@pytest.fixture
def cache_path(tmp_path):
    """Isolated cache file path — parent dir may or may not pre-exist."""
    return tmp_path / "jspb_templates.json"


@pytest.fixture
def templates():
    """Representative per-model templates matching 2026-04-21 live captures."""
    return {
        "pro": '[1,null,null,null,"797f3d0293f288ad",null,null,0,[4],null,null,3,null,null,3,1]',
        "flash": '[1,null,null,null,"56fdd199312815e2",null,null,0,[4],null,null,3,null,null,1]',
        "thinking": '[1,null,null,null,"aaaaaaaaaaaaaaaa",null,null,0,[4],null,null,3,null,null,2]',
    }


# ---------------------------------------------------------------------------
# 1. Roundtrip: write then read returns the same templates plus metadata.
# ---------------------------------------------------------------------------
def test_write_then_read_roundtrip(cache_path, templates):
    """A freshly-written cache reads back with identical templates and source.

    Validates the happy path: write_cache persists the payload and read_cache
    returns the structured record. Also confirms parent directory creation
    (tmp_path parent exists but deeper paths must be created by write_cache).
    """
    now = 1_745_256_000.0
    deep_path = cache_path.parent / "nested" / "jspb_templates.json"

    write_cache(templates, source="cdp", path=deep_path, now=now)

    record = read_cache(path=deep_path, now=now + 60.0)

    assert record is not None, "fresh cache should be readable immediately after write"
    assert record["captured_at"] == pytest.approx(now)
    assert record["source"] == "cdp"
    assert record["templates"] == templates
    assert is_cache_fresh(path=deep_path, now=now + 60.0) is True


# ---------------------------------------------------------------------------
# 2. Expiry: a cache older than 24h is treated as missing.
# ---------------------------------------------------------------------------
def test_read_returns_none_when_expired(cache_path, templates):
    """Cache captured more than 24h ago must not be returned.

    Default TTL is 24h. ``read_cache`` called with ``now`` past the TTL
    window returns None so the caller re-captures, and ``is_cache_fresh``
    agrees.
    """
    captured_at = 1_745_000_000.0
    write_cache(templates, source="cdp", path=cache_path, now=captured_at)

    expired_now = captured_at + DAY + 60.0

    assert read_cache(path=cache_path, now=expired_now) is None
    assert is_cache_fresh(path=cache_path, now=expired_now) is False


# ---------------------------------------------------------------------------
# 3. Invalidate past the debounce window actually removes the cache.
# ---------------------------------------------------------------------------
def test_invalidate_removes_fresh_cache_after_debounce(cache_path, templates):
    """After the debounce window elapses, invalidate_cache succeeds.

    The debounce is meant to block capture storms from non-drift failures,
    not to make invalidation permanently impossible. Once enough time has
    passed since the last write, a drift signal must be able to purge the
    cache so the next init re-captures.
    """
    captured_at = 1_745_000_000.0
    write_cache(templates, source="cdp", path=cache_path, now=captured_at)

    # 2h later — beyond the default 1h debounce window.
    later = captured_at + 2 * HOUR

    assert invalidate_cache(path=cache_path, now=later) is True
    assert read_cache(path=cache_path, now=later) is None
    assert is_cache_fresh(path=cache_path, now=later) is False


# ---------------------------------------------------------------------------
# 4. Invalidation within the debounce window is suppressed.
# ---------------------------------------------------------------------------
def test_invalidate_is_debounced_when_recent(cache_path, templates):
    """Invalidation within the debounce window is a no-op.

    Prevents a capture storm when a transient failure (quota exhaustion,
    rate limit, etc.) happens to surface as a PayloadValidationError soon
    after a fresh capture. Cache must be preserved and invalidate returns
    False to signal debounce.
    """
    captured_at = 1_745_000_000.0
    write_cache(templates, source="cdp", path=cache_path, now=captured_at)

    # 10 minutes later — well inside the default 1h debounce.
    soon = captured_at + 600.0

    assert invalidate_cache(path=cache_path, now=soon) is False
    record = read_cache(path=cache_path, now=soon)
    assert record is not None, "debounced invalidate must preserve the cache"
    assert record["templates"] == templates


# ---------------------------------------------------------------------------
# 5. Malformed JSON on disk: treated as cache miss, no exception bubbles up.
# ---------------------------------------------------------------------------
def test_read_returns_none_on_malformed_json(cache_path):
    """A corrupt cache file must not crash callers — treat as cache miss.

    Disk corruption or a partial write from a killed process must not
    propagate as an unhandled exception at import or request time. The
    contract is: read_cache returns None, is_cache_fresh is False, and
    invalidate_cache can still be called safely to clean up.
    """
    cache_path.write_text("{this is not valid json", encoding="utf-8")
    now = 1_745_000_000.0

    assert read_cache(path=cache_path, now=now) is None
    assert is_cache_fresh(path=cache_path, now=now) is False


# ---------------------------------------------------------------------------
# 6. Invalidate on a missing cache returns False without error.
# ---------------------------------------------------------------------------
def test_invalidate_missing_cache_returns_false(cache_path):
    """Invalidating a non-existent cache is a no-op and returns False.

    The return value signals 'nothing removed' so callers can avoid logging
    a spurious 'cache invalidated' message when there was nothing to purge.
    """
    assert not cache_path.exists()
    now = 1_745_000_000.0

    assert invalidate_cache(path=cache_path, now=now) is False
    assert read_cache(path=cache_path, now=now) is None


# ---------------------------------------------------------------------------
# 7. Schema validation: missing required keys → cache miss.
# ---------------------------------------------------------------------------
def test_read_returns_none_on_schema_violation(cache_path):
    """JSON parses but lacks required fields → treated as cache miss.

    A file with valid JSON but the wrong shape (e.g. missing ``templates``
    or ``captured_at``) must not be returned as a usable cache record.
    This can happen if a future schema version lands on disk and is then
    downgraded, or if a test fixture writes a partial record.
    """
    cache_path.write_text(
        json.dumps({"captured_at": 1_745_000_000.0, "source": "cdp"}),
        encoding="utf-8",
    )
    now = 1_745_000_000.0 + 60.0

    assert read_cache(path=cache_path, now=now) is None
    assert is_cache_fresh(path=cache_path, now=now) is False
