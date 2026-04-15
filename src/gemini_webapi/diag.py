"""python -m gemini_webapi.diag - on-demand payload drift diagnostic.

Captures a real Chrome StreamGenerate request via the WAA harvester and
diffs it against what the library would send for the same model. Prints
a slot-by-slot diff so maintainers can fix drift in one edit.

Pro model capture: Google's Pro entitlement only loads in Chrome profiles
with a completed Google account sign-in (not cookie injection alone).
To capture Pro reliably, start Chrome with a remote debugging port and
let the diag CLI connect via CDP:

    /Applications/Google\\ Chrome.app/Contents/MacOS/Google\\ Chrome \\
        --remote-debugging-port=9222 &
    python -m gemini_webapi.diag --model pro

Without CDP, the CLI falls back to a fresh Playwright Chrome, which
only reliably captures Flash (Pro/Thinking show as disabled because
account entitlement is missing from the fresh profile).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any
from urllib.request import urlopen

from .client import build_diagnostic_inner_req_list, _DIFF_EXCLUDED_SLOTS
from .constants import Model
from .utils.waa_token import harvest_waa_token, _parse_stream_generate_request


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


async def _capture_via_cdp(cdp_url: str, target_model: str) -> list | None:
    """Connect to an existing Chrome over CDP and capture a StreamGenerate
    request for the target model. Returns the full inner_req_list or None
    if capture failed.

    The existing Chrome must be signed in to the Google account used for
    Gemini. The connection shares cookies and entitlement state with the
    running browser, which is how Pro capture becomes possible.
    """
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        print("diag: playwright not installed", file=sys.stderr)
        return None

    captured: dict[str, Any] = {"inner": None}
    captured_event = asyncio.Event()

    async def handle_route(route):
        try:
            post_data = route.request.post_data
            if post_data:
                parsed = _parse_stream_generate_request(post_data)
                if parsed and isinstance(parsed["token"], str) and parsed["token"].startswith("!"):
                    captured["inner"] = parsed["reference_inner"]
                    captured_event.set()
        except Exception:
            pass
        await route.abort()

    label = _MODE_PICKER_LABEL.get(target_model, "Pro")
    try:
        async with async_playwright() as p:
            browser = await p.chromium.connect_over_cdp(cdp_url)
            if not browser.contexts:
                print("diag: CDP Chrome has no contexts", file=sys.stderr)
                return None
            ctx = browser.contexts[0]
            page = await ctx.new_page()
            try:
                await page.route("**/StreamGenerate*", handle_route)
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

                # Check current mode picker state. If not already the target,
                # open the picker and click the target option.
                current_label = await page.evaluate(
                    "() => { const b = document.querySelector('button[aria-label=\"Open mode picker\"]'); return b ? b.textContent.trim() : ''; }"
                )
                if current_label != label:
                    try:
                        await page.click(
                            'button[aria-label="Open mode picker"]', timeout=3000
                        )
                        await page.wait_for_selector(
                            f'[data-test-id="bard-mode-option-{target_model}"]',
                            timeout=3000,
                        )
                        await page.click(
                            f'[data-test-id="bard-mode-option-{target_model}"]',
                            timeout=3000,
                        )
                        await asyncio.sleep(0.5)
                    except Exception as e:
                        print(
                            f"diag: could not select mode {target_model!r}: {e}",
                            file=sys.stderr,
                        )
                        return None

                # Type warmup and send
                input_sel = 'div[contenteditable="true"], textarea'
                await page.click(input_sel, force=True)
                await page.type(input_sel, "hi")
                await page.click('button[aria-label*="Send"]', force=True, timeout=5000)
                try:
                    await asyncio.wait_for(captured_event.wait(), timeout=15)
                except asyncio.TimeoutError:
                    pass
            finally:
                try:
                    await page.close()
                except Exception:
                    pass
    except Exception as e:
        print(f"diag: CDP connection failed: {e}", file=sys.stderr)
        return None

    return captured["inner"]


def _cdp_port_open(cdp_url: str) -> bool:
    """Return True if a Chrome DevTools endpoint is reachable at cdp_url."""
    try:
        with urlopen(cdp_url + "/json/version", timeout=1) as resp:
            return resp.status == 200
    except Exception:
        return False


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
        default="http://localhost:9222",
        help="Chrome DevTools URL to connect to (default: http://localhost:9222). "
        "Preferred path for Pro/Thinking capture — requires Chrome launched "
        "with --remote-debugging-port and signed into the target Google account.",
    )
    parser.add_argument(
        "--no-cdp",
        action="store_true",
        help="Skip CDP and use the fresh-launch harvester (Flash-only reliably).",
    )
    args = parser.parse_args(argv)

    model_enum = _MODEL_ALIAS_TO_ENUM[args.model]

    reference_inner: list | None = None

    # Path A: try CDP to an existing signed-in Chrome first.
    if not args.no_cdp and _cdp_port_open(args.cdp_url):
        print(f"diag: connecting to Chrome via CDP at {args.cdp_url}", file=sys.stderr)
        reference_inner = await _capture_via_cdp(args.cdp_url, args.model)
        if reference_inner is None:
            print(
                "diag: CDP capture failed, falling back to fresh-launch harvester",
                file=sys.stderr,
            )

    # Path B: fall back to fresh Playwright launch (Flash-only reliably).
    if reference_inner is None:
        cookies = None
        if args.cookies and args.cookies.exists():
            cookies = _load_cookies(args.cookies)
        if cookies is None:
            # Provide an empty jar so the harvester doesn't crash on None.
            # Real harvests will fail with "no cookies"; tests mock the
            # harvester and never reach the cookie jar.
            from curl_cffi.requests import Cookies as _Cookies
            cookies = _Cookies()
        result = await harvest_waa_token(cookies, target_model=args.model)
        if not isinstance(result, tuple) or len(result) < 5:
            print(
                "diag: harvest_waa_token did not return a 5-tuple (legacy?)",
                file=sys.stderr,
            )
            return 2
        reference_inner = result[4]
    if reference_inner is None:
        print(
            "diag: no reference inner list captured.",
            file=sys.stderr,
        )
        return 2

    built = build_diagnostic_inner_req_list(model_enum)
    drifts = _diff_slots(built, reference_inner, _DIFF_EXCLUDED_SLOTS)

    if not drifts:
        print(f"No drift detected for model={args.model}.")
        return 0

    print(f"Drift detected for model={args.model}:")
    for d in drifts:
        print(
            f"  slot={d['position']} "
            f"client={d['client_value']!r} "
            f"chrome={d['chrome_value']!r} "
            f"kind={d['kind']}"
        )
    return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
