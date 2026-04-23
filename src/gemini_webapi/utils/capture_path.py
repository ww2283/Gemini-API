"""CDP probe + capture path resolver for per-model jspb template capture."""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import Callable


MANAGED_PROFILE_DIR = Path("~/.cache/gemini_webapi/chrome_profile").expanduser()
MANAGED_CDP_PORT = 9423


def probe_cdp_url(url: str, timeout_seconds: float = 0.3) -> bool:
    """True iff Chrome responds at {url}/json/version with Browser starts-with 'Chrome/'."""
    try:
        req = urllib.request.Request(f"{url}/json/version")
        with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        browser = data.get("Browser")
        return isinstance(browser, str) and browser.startswith("Chrome/")
    except Exception:
        return False


def resolve_capture_path(
    cdp_url_override: str | None = None,
    managed_profile_path: Path | None = None,
    probe_fn: Callable[[str, float], bool] | None = None,
) -> tuple[str, str | None]:
    """Pick transport in order CDP > managed > fresh; return (source, url_or_path)."""
    probe = probe_fn if probe_fn is not None else probe_cdp_url
    url = cdp_url_override or "http://localhost:9222"
    if probe(url, 0.3):
        return ("cdp", url)
    if (
        managed_profile_path is not None
        and managed_profile_path.exists()
        and (managed_profile_path / "Default").exists()
    ):
        return ("managed", str(managed_profile_path))
    return ("fresh", None)
