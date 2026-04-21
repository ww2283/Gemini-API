"""Tests for gemini_webapi.utils.capture_path -- CDP probe + capture path resolver.

R3 Red phase: pins the public contract of the not-yet-implemented
``capture_path`` module. These tests must fail at import time until G3
creates ``src/gemini_webapi/utils/capture_path.py``.

Contract under test (from .doc/plans/2026-04-21_per_model_template_capture.md):

- ``probe_cdp_url``        : non-blocking probe of a CDP endpoint; True iff
                             ``{url}/json/version`` returns 200 JSON with a
                             "Browser" key starting with "Chrome/". Never raises.
- ``resolve_capture_path`` : picks transport in order CDP > managed > fresh.
                             Returns ``(source, url_or_path)``.

``probe_fn`` is injected into ``resolve_capture_path`` so resolver tests never
touch real sockets. Only ``test_probe_cdp_url_returns_false_on_closed_port``
exercises actual network behavior, and it targets a port known to be closed.
"""
from __future__ import annotations

import pytest

from gemini_webapi.utils.capture_path import (
    probe_cdp_url,
    resolve_capture_path,
)


# ---------------------------------------------------------------------------
# probe_cdp_url -- real network behavior on a closed port
# ---------------------------------------------------------------------------


def test_probe_cdp_url_returns_false_on_closed_port():
    """Probe against a port that is definitely not listening.

    Port 1 on loopback is reserved (tcpmux) and essentially never open on
    developer machines. The probe must return False without raising -- any
    socket error, timeout, or connection refusal is a False result, not an
    exception, because callers treat it as a simple boolean gate.
    """
    result = probe_cdp_url("http://127.0.0.1:1", timeout_seconds=0.3)
    assert result is False


# ---------------------------------------------------------------------------
# resolve_capture_path -- transport selection
# ---------------------------------------------------------------------------


def test_resolve_prefers_cdp_when_probe_succeeds(tmp_path):
    """CDP wins over a valid managed profile when the probe returns True.

    Precedence is: CDP > managed > fresh. Even when managed_profile_path is
    set AND exists AND has a Default subdir, a successful CDP probe short-
    circuits resolution -- we do not want to spin up a managed browser when
    a user already has Chrome running with --remote-debugging-port.
    """
    # Managed profile is fully valid -- but CDP should win anyway.
    (tmp_path / "Default").mkdir()

    source, url_or_path = resolve_capture_path(
        cdp_url_override=None,
        managed_profile_path=tmp_path,
        probe_fn=lambda url, timeout: True,
    )

    assert source == "cdp"
    assert url_or_path == "http://localhost:9222"


def test_resolve_falls_through_to_managed_when_cdp_probe_fails(tmp_path):
    """Managed profile wins when CDP probe fails AND profile looks set up.

    "Looks set up" means the profile directory exists AND contains a
    ``Default`` subdirectory -- Chrome creates ``Default`` on first run, so
    its presence is a proxy for "``--setup`` was already run".
    """
    (tmp_path / "Default").mkdir()

    source, url_or_path = resolve_capture_path(
        cdp_url_override=None,
        managed_profile_path=tmp_path,
        probe_fn=lambda url, timeout: False,
    )

    assert source == "managed"
    assert url_or_path == str(tmp_path)


def test_resolve_falls_through_to_fresh_when_no_managed_profile():
    """Fresh launch is the last-resort fallback when nothing else works.

    When CDP probe fails AND no managed profile path is provided, the
    resolver must fall through to ("fresh", None). The caller uses this to
    trigger a Playwright fresh-launch harvester, which is Flash-only.
    """
    source, url_or_path = resolve_capture_path(
        cdp_url_override=None,
        managed_profile_path=None,
        probe_fn=lambda url, timeout: False,
    )

    assert source == "fresh"
    assert url_or_path is None


def test_resolve_uses_cdp_url_override_when_provided():
    """The override URL is tried instead of the default localhost:9222.

    ``cdp_url_override`` lets the caller honor ``$GEMINI_WAA_CHROME_URL`` or a
    managed profile that launched on a nondefault port (diag uses 9423).
    The probe must be called with the override URL, and the returned url
    must match -- not silently fall back to the default.
    """
    probed_urls: list[str] = []

    def probe(url: str, timeout: float) -> bool:
        probed_urls.append(url)
        return url == "http://localhost:9423"

    source, url_or_path = resolve_capture_path(
        cdp_url_override="http://localhost:9423",
        managed_profile_path=None,
        probe_fn=probe,
    )

    assert source == "cdp"
    assert url_or_path == "http://localhost:9423"
    assert "http://localhost:9423" in probed_urls


def test_resolve_rejects_managed_profile_without_Default_subdir(tmp_path):
    """Profile dir without ``Default`` is treated as "setup never ran".

    A user could create the cache directory by hand, or a partial/aborted
    ``--setup`` run could leave an empty directory. We must not try to
    launch Chrome against an empty profile -- that would silently create a
    fresh unsigned-in profile, which is the failure mode ``--setup``
    exists to prevent. Fall through to fresh instead.
    """
    # tmp_path exists but has no Default subdirectory.
    assert tmp_path.exists()
    assert not (tmp_path / "Default").exists()

    source, url_or_path = resolve_capture_path(
        cdp_url_override=None,
        managed_profile_path=tmp_path,
        probe_fn=lambda url, timeout: False,
    )

    assert source == "fresh"
    assert url_or_path is None
