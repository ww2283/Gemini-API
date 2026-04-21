"""python -m gemini_webapi.diag - on-demand payload drift diagnostic.

This is an opt-in maintainer tool. It does NOT run automatically during
gemini-webapi requests, and it does NOT consume cookies from the daily
workflow's cookie store. Run it only when you suspect payload drift.

Chrome connection strategy
--------------------------

Pro/Thinking entitlement only loads in Chrome profiles with a completed
Google account sign-in — cookie injection alone leaves those modes
``aria-disabled``. The diag CLI connects to a signed-in Chrome via the
DevTools Protocol and reuses that browser's context.

1. If a Chrome instance is already listening on ``--cdp-url``
   (default ``http://localhost:9222``), diag connects to it. Useful if
   you manually start Chrome with ``--remote-debugging-port=9222``.
2. Otherwise, diag uses a **managed Chrome profile** it owns at
   ``~/.cache/gemini_webapi/chrome_profile``. It launches Chrome against
   that profile with a private debug port, then connects via CDP.
3. The first time you use the managed profile you must run
   ``python -m gemini_webapi.diag --setup`` — this opens Chrome headful,
   navigates to gemini.google.com, and waits for you to sign in to your
   Google account. The sign-in state persists in the managed profile
   directory; subsequent runs are automatic.
4. The fresh-launch harvester path (``--no-cdp``) falls back to
   ``harvest_waa_token``. It only reliably captures Flash because
   fresh Playwright contexts lack Pro entitlement.

Do NOT run diag on a schedule or in a tight loop. Repeated fresh Chrome
sessions can trigger Google's abuse detection, which forces a
sign-out/sign-in cycle that breaks the main cookie-extraction workflow.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from urllib.request import urlopen

from .client import build_diagnostic_inner_req_list, _DIFF_EXCLUDED_SLOTS
from .constants import Model
from .utils.template_capture import (
    _TRACKED_HEADERS,
    capture_model_template,
)
from .utils.waa_token import (
    harvest_waa_token,
    _find_system_chrome,
)


_MODEL_ALIAS_TO_ENUM: dict[str, Model] = {
    "pro": Model.G_3_1_PRO,
    "flash": Model.G_3_0_FLASH,
    "thinking": Model.G_3_0_FLASH_THINKING,
}

_MODE_PICKER_LABEL: dict[str, str] = {
    "pro": "Pro",
    "flash": "Fast",
    "thinking": "Thinking",
}

_DEFAULT_CDP_URL = "http://localhost:9222"
_MANAGED_PROFILE_DIR = Path("~/.cache/gemini_webapi/chrome_profile").expanduser()
_MANAGED_CDP_PORT = 9423  # Fixed, unlikely to clash with user tools


def _managed_profile_exists() -> bool:
    """Return True if the managed profile directory has signed-in state."""
    return (_MANAGED_PROFILE_DIR / "Default" / "Login Data For Account").exists() or (
        _MANAGED_PROFILE_DIR / "Default" / "Cookies"
    ).exists()


def _cdp_url_ready(cdp_url: str, timeout: float = 1.0) -> bool:
    """Return True if a Chrome DevTools endpoint responds at cdp_url."""
    try:
        with urlopen(cdp_url + "/json/version", timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False


def _launch_managed_chrome(headless: bool) -> subprocess.Popen | None:
    """Launch the managed Chrome profile with a dedicated debug port.

    Returns the process handle on success, None if Chrome cannot be found.
    Caller is responsible for terminating the process when done.
    """
    chrome_path = _find_system_chrome()
    if not chrome_path:
        print("diag: system Chrome not found.", file=sys.stderr)
        return None
    _MANAGED_PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    args = [
        chrome_path,
        f"--user-data-dir={_MANAGED_PROFILE_DIR}",
        f"--remote-debugging-port={_MANAGED_CDP_PORT}",
        "--no-first-run",
        "--no-default-browser-check",
    ]
    if headless:
        # --headless=new is required to avoid "HeadlessChrome" UA detection.
        args.append("--headless=new")
        # Keep windows off-screen just in case.
        args.append("--window-position=-10000,-10000")
        args.append("--window-size=1,1")
    proc = subprocess.Popen(
        args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    # Wait for the debug port to come up (up to 10s).
    managed_url = f"http://localhost:{_MANAGED_CDP_PORT}"
    for _ in range(50):
        if _cdp_url_ready(managed_url, timeout=0.3):
            return proc
        time.sleep(0.2)
    # Port never opened.
    proc.terminate()
    return None


async def _capture_via_cdp(
    cdp_url: str, target_model: str
) -> dict[str, Any] | None:
    """Connect to a Chrome at cdp_url and capture a StreamGenerate request
    for target_model.

    Returns ``{"inner": list, "headers": {header_name: value},
    "raw_jspb": str}`` on success or ``None`` on failure. ``headers`` only
    includes values that were present on the captured request; missing
    entries are omitted.
    """
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        print("diag: playwright not installed", file=sys.stderr)
        return None

    try:
        async with async_playwright() as p:
            browser = await p.chromium.connect_over_cdp(cdp_url)
            if not browser.contexts:
                print("diag: CDP Chrome has no contexts", file=sys.stderr)
                return None
            ctx = browser.contexts[0]
            page = await ctx.new_page()
            try:
                await page.goto(
                    "https://gemini.google.com/app",
                    wait_until="domcontentloaded",
                    timeout=45000,
                )
                await page.wait_for_selector(
                    'div[contenteditable="true"], textarea', timeout=20000
                )
                await asyncio.sleep(1)

                # Dismiss the "Supercharge Gemini with Personal Intelligence"
                # onboarding modal if present.
                try:
                    await page.evaluate(
                        """() => {
                            for (const btn of document.querySelectorAll('button')) {
                                if (btn.textContent.trim() === 'Not now') btn.click();
                            }
                        }"""
                    )
                    await asyncio.sleep(0.3)
                except Exception:
                    pass

                return await capture_model_template(page, target_model)
            finally:
                try:
                    await page.close()
                except Exception:
                    pass
    except Exception as e:
        print(f"diag: CDP connection failed: {e}", file=sys.stderr)
        return None


def _load_cookies(path: Path) -> Any:
    """Load cookies from a JSON file. Returns a curl_cffi Cookies jar."""
    from curl_cffi.requests import Cookies

    raw = json.loads(path.read_text())
    jar = Cookies()
    for name, value in raw.items():
        if value:
            jar.set(name, value, domain=".google.com")
    return jar


def _diff_slots(
    built: list, reference: list, excluded: frozenset[int]
) -> list[dict]:
    """Pure slot-by-slot diff, same logic as GeminiClient._diff_inner_req_list."""
    entries: list[dict] = []
    for position in range(min(len(built), len(reference))):
        if position in excluded:
            continue
        c = built[position]
        r = reference[position]
        if r is None:
            continue
        if c is None:
            entries.append(
                {
                    "position": position,
                    "client_value": None,
                    "chrome_value": r,
                    "kind": "missing_in_client",
                }
            )
        elif c != r:
            entries.append(
                {
                    "position": position,
                    "client_value": c,
                    "chrome_value": r,
                    "kind": "value_mismatch",
                }
            )
    return entries


def _parse_jspb_array(value: str) -> list | None:
    """Parse a jspb header value as a JSON array. Returns None if malformed."""
    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return None
    return parsed if isinstance(parsed, list) else None


def _diff_headers(
    model_enum: Model, captured: dict[str, str]
) -> list[dict]:
    """Diff the library's model headers against a Chrome capture.

    For each tracked jspb header: if both sides parse as JSON arrays, reports
    slot-by-slot mismatches. If only one side has a value or the values fail
    to parse, reports the header-level mismatch.

    The library-side model_id (slot 4 of the 525001261 header) is excluded
    from the diff because the Model enum pins it to one of several valid
    account-specific IDs and Chrome may have rotated to a different one.
    """
    lib_headers = model_enum.model_header
    entries: list[dict] = []
    for hname in _TRACKED_HEADERS:
        lib_value = lib_headers.get(hname)
        chrome_value = captured.get(hname)
        if chrome_value is None:
            # No Chrome reference for this header — skip silently rather
            # than flagging drift. The fresh-launch harvester path never
            # captures headers, so a missing value is not a bug.
            continue
        if lib_value is None:
            entries.append(
                {
                    "header": hname,
                    "client_value": None,
                    "chrome_value": chrome_value,
                    "kind": "missing_in_client",
                }
            )
            continue
        lib_slots = _parse_jspb_array(lib_value)
        chrome_slots = _parse_jspb_array(chrome_value)
        if lib_slots is None or chrome_slots is None:
            if lib_value != chrome_value:
                entries.append(
                    {
                        "header": hname,
                        "client_value": lib_value,
                        "chrome_value": chrome_value,
                        "kind": "string_mismatch",
                    }
                )
            continue
        # Compare up to the longer length so missing trailing slots surface.
        for position in range(max(len(lib_slots), len(chrome_slots))):
            # Slot 4 is the account-specific model_id; any library pin for
            # a valid account is legitimate so skip.
            if hname == "x-goog-ext-525001261-jspb" and position == 4:
                continue
            l = lib_slots[position] if position < len(lib_slots) else None
            r = chrome_slots[position] if position < len(chrome_slots) else None
            if l != r:
                entries.append(
                    {
                        "header": hname,
                        "position": position,
                        "client_value": l,
                        "chrome_value": r,
                        "kind": "slot_mismatch",
                    }
                )
    return entries


def _run_setup() -> int:
    """Launch the managed Chrome profile headful so the user can sign in.

    Returns an exit code.
    """
    chrome_path = _find_system_chrome()
    if not chrome_path:
        print("diag: system Chrome not found.", file=sys.stderr)
        return 2
    _MANAGED_PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    print(
        "diag: launching Chrome with the managed profile at\n"
        f"      {_MANAGED_PROFILE_DIR}\n"
        "      Sign in to your Google account in the Chrome window.\n"
        "      When Gemini shows your account is signed in, close the\n"
        "      Chrome window to complete setup.",
        file=sys.stderr,
    )
    args = [
        chrome_path,
        f"--user-data-dir={_MANAGED_PROFILE_DIR}",
        "--no-first-run",
        "--no-default-browser-check",
        "https://gemini.google.com/app",
    ]
    proc = subprocess.Popen(args)
    proc.wait()
    if _managed_profile_exists():
        print("diag: managed profile initialized.", file=sys.stderr)
        return 0
    print(
        "diag: managed profile did not retain sign-in state.",
        file=sys.stderr,
    )
    return 1


def _reset_managed_profile() -> int:
    if _MANAGED_PROFILE_DIR.exists():
        shutil.rmtree(_MANAGED_PROFILE_DIR)
        print(f"diag: removed {_MANAGED_PROFILE_DIR}", file=sys.stderr)
    return 0


async def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m gemini_webapi.diag")
    parser.add_argument(
        "--model",
        choices=list(_MODEL_ALIAS_TO_ENUM.keys()),
        default="pro",
        help="Target model alias (pro|flash|thinking).",
    )
    parser.add_argument(
        "--cookies",
        type=Path,
        default=None,
        help="Path to a JSON cookies file (fallback path only).",
    )
    parser.add_argument(
        "--cdp-url",
        type=str,
        default=_DEFAULT_CDP_URL,
        help="External Chrome DevTools URL to try first (default: "
        "http://localhost:9222). Optional — the managed profile is used "
        "if no external debug port is open.",
    )
    parser.add_argument(
        "--no-cdp",
        action="store_true",
        help="Skip CDP and use the fresh-launch harvester (Flash-only reliably).",
    )
    parser.add_argument(
        "--setup",
        action="store_true",
        help="Launch the managed Chrome profile headful so you can sign in "
        "to your Google account. Run once before the first diag capture.",
    )
    parser.add_argument(
        "--reset-profile",
        action="store_true",
        help="Delete the managed Chrome profile directory.",
    )
    parser.add_argument(
        "--managed-headful",
        action="store_true",
        help="Run the managed Chrome profile headful instead of headless "
        "during capture (visible window, useful for debugging).",
    )
    args = parser.parse_args(argv)

    if args.reset_profile:
        return _reset_managed_profile()
    if args.setup:
        return _run_setup()

    model_enum = _MODEL_ALIAS_TO_ENUM[args.model]
    reference_inner: list | None = None
    reference_headers: dict[str, str] = {}
    managed_proc: subprocess.Popen | None = None

    def _absorb_capture(capture: dict[str, Any] | None) -> list | None:
        """Unpack CDP capture dict into module-local reference state."""
        if capture is None:
            return None
        inner = capture.get("inner")
        headers = capture.get("headers") or {}
        if isinstance(headers, dict):
            for k, v in headers.items():
                if isinstance(v, str):
                    reference_headers[k] = v
        return inner if isinstance(inner, list) else None

    try:
        # Path A: external CDP first (maintainer has Chrome running with debug port).
        if not args.no_cdp and _cdp_url_ready(args.cdp_url):
            print(
                f"diag: connecting to external Chrome via CDP at {args.cdp_url}",
                file=sys.stderr,
            )
            reference_inner = _absorb_capture(
                await _capture_via_cdp(args.cdp_url, args.model)
            )

        # Path B: managed profile. Launch Chrome ourselves with a dedicated
        # port and user-data-dir, then connect via CDP.
        if reference_inner is None and not args.no_cdp:
            if not _managed_profile_exists():
                print(
                    "diag: no managed profile found. Run "
                    "'python -m gemini_webapi.diag --setup' first to sign "
                    "in to your Google account.",
                    file=sys.stderr,
                )
            else:
                print(
                    "diag: launching managed Chrome profile "
                    f"({'headful' if args.managed_headful else 'headless'})",
                    file=sys.stderr,
                )
                managed_proc = _launch_managed_chrome(
                    headless=not args.managed_headful
                )
                if managed_proc is not None:
                    managed_url = f"http://localhost:{_MANAGED_CDP_PORT}"
                    reference_inner = _absorb_capture(
                        await _capture_via_cdp(managed_url, args.model)
                    )

        # Path C: fresh-launch fallback (Flash-only reliably).
        if reference_inner is None:
            cookies = None
            if args.cookies and args.cookies.exists():
                cookies = _load_cookies(args.cookies)
            if cookies is None:
                from curl_cffi.requests import Cookies as _Cookies
                cookies = _Cookies()
            print(
                "diag: falling back to fresh-launch harvester "
                "(Pro/Thinking will likely capture as Flash, no headers)",
                file=sys.stderr,
            )
            result = await harvest_waa_token(cookies)
            if not isinstance(result, tuple) or len(result) < 5:
                print(
                    "diag: harvest_waa_token did not return a 5-tuple",
                    file=sys.stderr,
                )
                return 2
            harvested = result[4]
            reference_inner = harvested if isinstance(harvested, list) else None

        if reference_inner is None:
            print("diag: no reference inner list captured.", file=sys.stderr)
            return 2

        built = build_diagnostic_inner_req_list(model_enum)
        body_drifts = _diff_slots(built, reference_inner, _DIFF_EXCLUDED_SLOTS)
        header_drifts = _diff_headers(model_enum, reference_headers)

        any_drift = bool(body_drifts) or bool(header_drifts)

        if body_drifts:
            print(f"Body drift detected for model={args.model}:")
            for d in body_drifts:
                print(
                    f"  slot={d['position']} "
                    f"client={d['client_value']!r} "
                    f"chrome={d['chrome_value']!r} "
                    f"kind={d['kind']}"
                )

        if header_drifts:
            print(f"Header drift detected for model={args.model}:")
            for d in header_drifts:
                if d["kind"] == "slot_mismatch":
                    print(
                        f"  header={d['header']} slot={d['position']} "
                        f"client={d['client_value']!r} "
                        f"chrome={d['chrome_value']!r}"
                    )
                else:
                    print(
                        f"  header={d['header']} "
                        f"client={d.get('client_value')!r} "
                        f"chrome={d.get('chrome_value')!r} "
                        f"kind={d['kind']}"
                    )
        elif reference_headers:
            print(
                f"No header drift detected for model={args.model} "
                f"(compared: {', '.join(sorted(reference_headers))})."
            )
        else:
            print(
                f"No headers captured for model={args.model}; header diff "
                "skipped (CDP capture required).",
                file=sys.stderr,
            )

        if not any_drift:
            if not body_drifts:
                print(f"No drift detected for model={args.model}.")
            return 0
        return 1

    finally:
        if managed_proc is not None:
            try:
                managed_proc.terminate()
                managed_proc.wait(timeout=5)
            except Exception:
                try:
                    managed_proc.kill()
                except Exception:
                    pass


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
