"""Auto-capture and hotpatch for ``x-goog-ext-525001261-jspb`` drift.

The jspb header is the tightest coupling point between the library and
Google's server-side schema. When Google changes which slots are required
to be null, the library's hardcoded values in :mod:`constants` drift and
long-stream Pro requests silently truncate.

This module extracts a fresh reference header from the WAA harvester's
capture (which already runs on every :meth:`GeminiClient.init` when
``waa_token_provider=True``), sanity-checks it, and hotpatches the
in-memory :class:`Model` enum so subsequent requests use the current
server-accepted structure.

The patch is additive — if validation fails the library silently falls
back to the static :mod:`constants` values, so auto-patch can never make
the client less functional than it was without this module.

Opt-in persistence (``GEMINI_JSPB_AUTOPATCH_CACHE=1``) writes the captured
structure to ``~/.cache/gemini_webapi/jspb_template.json`` so subsequent
inits that skip the full WAA harvest can still load a recent template.
The cache has a 24h TTL and no manual invalidation is required.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from loguru import logger

from ..constants import Model

_JSPB_HEADER_NAME = "x-goog-ext-525001261-jspb"
_EXPECTED_SLOT_COUNT = 15
_MODEL_ID_SLOT = 4
_GUARD_SLOT = 8  # Must be [4]
_VARIANT_SLOT = 14

_CACHE_PATH = Path("~/.cache/gemini_webapi/jspb_template.json").expanduser()
_CACHE_TTL_SECONDS = 24 * 60 * 60


def _parse_jspb(value: str) -> list | None:
    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return None
    return parsed if isinstance(parsed, list) else None


def _is_valid_template(slots: list) -> bool:
    """Structural sanity checks on a captured jspb header.

    A valid Google-side template has:
    - exactly 15 slots
    - slot 0 = 1
    - slot 4 = str (model_id, hex)
    - slot 8 = [4]
    - slot 14 = int (variant)

    Everything else may legitimately be null or vary. Fail-closed: any
    mismatch causes the caller to skip hotpatching and keep the static
    constants.
    """
    if len(slots) != _EXPECTED_SLOT_COUNT:
        return False
    if slots[0] != 1:
        return False
    model_id = slots[_MODEL_ID_SLOT]
    if not isinstance(model_id, str) or len(model_id) != 16:
        return False
    if slots[_GUARD_SLOT] != [4]:
        return False
    variant = slots[_VARIANT_SLOT]
    if not isinstance(variant, int):
        return False
    return True


def _derive_model_header(
    template: list, target_model: Model
) -> str | None:
    """Substitute ``target_model``'s model_id and variant into ``template``.

    The captured template carries the slot structure Google currently
    accepts (which slots are null, which carry values). All three models
    share that structure and differ only in slot 4 (model_id) and slot 14
    (variant). By copying the template and swapping those two slots, one
    Flash capture is enough to patch Pro, Flash, and Thinking.

    Returns the JSON-encoded header string, or ``None`` if the static
    model header is malformed (which would mean a deeper bug — don't
    patch in that case).
    """
    static_header_str = target_model.model_header.get(_JSPB_HEADER_NAME)
    if not isinstance(static_header_str, str):
        return None
    static_slots = _parse_jspb(static_header_str)
    if static_slots is None or len(static_slots) <= _VARIANT_SLOT:
        return None
    static_model_id = static_slots[_MODEL_ID_SLOT]
    static_variant = static_slots[_VARIANT_SLOT]
    if not isinstance(static_model_id, str) or not isinstance(static_variant, int):
        return None
    patched = list(template)
    patched[_MODEL_ID_SLOT] = static_model_id
    patched[_VARIANT_SLOT] = static_variant
    # Emit without spaces to match Chrome's wire format exactly.
    return json.dumps(patched, separators=(",", ":"))


def _cache_enabled() -> bool:
    return os.environ.get("GEMINI_JSPB_AUTOPATCH_CACHE") == "1"


def _load_cached_template() -> list | None:
    """Return the cached template if it exists and is fresh, else None."""
    if not _cache_enabled() or not _CACHE_PATH.exists():
        return None
    try:
        data = json.loads(_CACHE_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    ts = data.get("captured_at")
    template = data.get("template")
    if not isinstance(ts, (int, float)) or not isinstance(template, list):
        return None
    if time.time() - ts > _CACHE_TTL_SECONDS:
        return None
    return template if _is_valid_template(template) else None


def _save_cached_template(template: list) -> None:
    if not _cache_enabled():
        return
    try:
        _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _CACHE_PATH.write_text(
            json.dumps({"captured_at": time.time(), "template": template})
        )
    except OSError as err:
        logger.debug(f"jspb_patch: could not write cache: {err}")


def apply_autopatch(reference_headers: dict[str, str] | None) -> int:
    """Hotpatch :class:`Model` enum headers from a captured reference.

    Call this after the WAA harvester returns. Extracts the template from
    the captured ``x-goog-ext-525001261-jspb`` value, validates it, and
    rewrites each :class:`Model` enum's ``model_header`` dict in place.

    Returns the number of models whose header was changed (0 = no patch
    needed or capture missing/invalid; 3 = all three models patched).

    The function never raises. Any error causes the static constants
    to remain unchanged.
    """
    template: list | None = None

    if reference_headers:
        raw = reference_headers.get(_JSPB_HEADER_NAME)
        if isinstance(raw, str):
            parsed = _parse_jspb(raw)
            if parsed is not None and _is_valid_template(parsed):
                template = parsed
                _save_cached_template(parsed)

    if template is None:
        template = _load_cached_template()

    if template is None:
        return 0

    patched_count = 0
    for model_enum in (
        Model.G_3_1_PRO,
        Model.G_3_0_FLASH,
        Model.G_3_0_FLASH_THINKING,
    ):
        derived = _derive_model_header(template, model_enum)
        if derived is None:
            continue
        current = model_enum.model_header.get(_JSPB_HEADER_NAME)
        if derived == current:
            continue
        model_enum.model_header[_JSPB_HEADER_NAME] = derived
        logger.warning(
            f"jspb_patch: hotpatched {model_enum.name} header "
            f"(was {current!r}, now {derived!r})"
        )
        patched_count += 1
    return patched_count
