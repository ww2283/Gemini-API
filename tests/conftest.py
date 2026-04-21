"""
Global test fixtures.

CDP probe neutralization: ``gemini_webapi.utils.capture_path.probe_cdp_url``
hits localhost:9222 with a 300ms timeout. If the developer happens to be
running Chrome with a debug port open (common for chrome-devtools MCP or
interactive debugging), the probe succeeds and tests that exercise
``harvest_waa_token``'s fresh-launch path pick up the CDP branch instead,
where their fakes don't support ``connect_over_cdp``.

Forcing the probe to return ``False`` at the autouse layer keeps those
tests deterministic regardless of the host Chrome state. R7 tests that
exercise the CDP path patch ``resolve_capture_path`` directly, so this
fixture does not interfere with them.
"""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _force_cdp_probe_false(monkeypatch):
    from gemini_webapi.utils import capture_path

    monkeypatch.setattr(
        capture_path,
        "probe_cdp_url",
        lambda *_a, **_kw: False,
    )
