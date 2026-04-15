"""python -m gemini_webapi.diag - on-demand payload drift diagnostic.

Captures a real Chrome StreamGenerate request via the WAA harvester and
diffs it against what the library would send for the same model. Prints
a slot-by-slot diff so maintainers can fix drift in one edit.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from .client import build_diagnostic_inner_req_list, _DIFF_EXCLUDED_SLOTS
from .constants import Model
from .utils.waa_token import harvest_waa_token


_MODEL_ALIAS_TO_ENUM: dict[str, Model] = {
    "pro": Model.G_3_1_PRO,
    "flash": Model.G_3_0_FLASH,
    "thinking": Model.G_3_0_FLASH_THINKING,
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
        help="Path to a JSON cookies file.",
    )
    args = parser.parse_args(argv)

    model_enum = _MODEL_ALIAS_TO_ENUM[args.model]

    cookies = None
    if args.cookies and args.cookies.exists():
        cookies = _load_cookies(args.cookies)

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
            "diag: harvester did not capture a reference inner list.",
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
