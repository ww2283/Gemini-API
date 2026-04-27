"""Auto-capture and hotpatch for ``x-goog-ext-525001261-jspb`` drift.

The jspb header is the tightest coupling point between the library and
Google's server-side schema. When Google changes which slots are required
to be null, the library's hardcoded values in :mod:`constants` drift and
long-stream Pro requests silently truncate.

This module takes per-model captured jspb header strings (one per
``Model`` enum) and hotpatches each ``Model.model_header`` dict in place.
Each template is validated independently; a bad template for one model
does not prevent the others from being patched.

The patch is additive -- if validation fails the library silently falls
back to the static :mod:`constants` values, so auto-patch can never make
the client less functional than it was without this module.

Cache persistence is the responsibility of :mod:`.jspb_cache`; this
module no longer touches disk.
"""
from __future__ import annotations

import json

from loguru import logger

from ..constants import Model

_JSPB_HEADER_NAME = "x-goog-ext-525001261-jspb"
_GUARD_SLOT = 8  # Must be [4] when present

_MODEL_KEY_MAP: dict[str, Model] = {
    "pro": Model.G_3_1_PRO,
    "flash": Model.G_3_0_FLASH,
    "thinking": Model.G_3_0_FLASH_THINKING,
}


def _parse_jspb(value: str) -> list | None:
    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return None
    return parsed if isinstance(parsed, list) else None


def _is_valid_template(slots: list) -> bool:
    """Permissive structural validator for live-captured jspb templates.

    The harvester captures values from StreamGenerate requests that the
    server actually accepted, so the captured value is by definition
    wire-valid. The validator's job is only to catch capture errors
    (empty/wrong header) and protocol-level shifts, not to model
    Google's per-slot schema.
    """
    if not slots:
        return False
    if slots[0] != 1:
        return False
    if len(slots) > _GUARD_SLOT and slots[_GUARD_SLOT] != [4]:
        return False
    return True


def apply_autopatch(templates: dict[str, str] | None) -> int:
    """Patch each Model enum's jspb header from its own captured template.

    ``templates`` maps canonical model keys ("pro", "flash", "thinking") to
    raw jspb header strings captured from live Chrome. Each template is
    validated independently; invalid templates are skipped without
    affecting the others. The raw string is written verbatim onto the
    corresponding ``Model.model_header`` dict -- no derivation, no slot
    substitution, no re-encoding.

    Returns the number of model headers that actually changed. Unknown
    keys are ignored silently. ``None`` or empty input returns 0.

    The function never raises. Any error causes the static constants
    to remain unchanged for that particular model.
    """
    if not templates:
        return 0

    patched_count = 0
    for key, raw in templates.items():
        model_enum = _MODEL_KEY_MAP.get(key)
        if model_enum is None:
            continue
        if not isinstance(raw, str):
            continue
        parsed = _parse_jspb(raw)
        if parsed is None or not _is_valid_template(parsed):
            continue
        current = model_enum.model_header.get(_JSPB_HEADER_NAME)
        if raw == current:
            continue
        model_enum.model_header[_JSPB_HEADER_NAME] = raw
        logger.warning(
            f"jspb_patch: hotpatched {model_enum.name} header "
            f"(was {current!r}, now {raw!r})"
        )
        patched_count += 1
    return patched_count
