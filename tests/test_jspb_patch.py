"""Tests for gemini_webapi.utils.jspb_patch -- the auto-capture hotpatch."""
from __future__ import annotations

import json

import pytest

from gemini_webapi.constants import Model
from gemini_webapi.utils.jspb_patch import (
    _is_valid_template,
    _parse_jspb,
    apply_autopatch,
)

_JSPB = "x-goog-ext-525001261-jspb"


@pytest.fixture(autouse=True)
def restore_model_headers():
    """Snapshot and restore model headers around each test.

    apply_autopatch mutates the Model enum dicts in place; without this
    fixture a test that patches Pro bleeds into every subsequent test.
    """
    snapshot = {m: dict(m.model_header) for m in Model if m is not Model.UNSPECIFIED}
    try:
        yield
    finally:
        for m, header in snapshot.items():
            m.model_header.clear()
            m.model_header.update(header)


def test_parse_jspb_returns_list_for_valid_json():
    assert _parse_jspb("[1,2,null]") == [1, 2, None]


def test_parse_jspb_returns_none_for_malformed():
    assert _parse_jspb("{not json}") is None
    assert _parse_jspb("") is None


def test_parse_jspb_rejects_non_list():
    assert _parse_jspb('{"a": 1}') is None
    assert _parse_jspb("42") is None


def test_valid_template_accepts_canonical_flash_shape():
    template = [
        1, None, None, None, "fbb127bbb056c959",
        None, None, None, [4], None, None,
        None, None, None, 1,
    ]
    assert _is_valid_template(template) is True


def test_valid_template_rejects_too_few_slots():
    """Templates with fewer than 15 slots are invalid."""
    template = [1, None, None, None, "fbb127bbb056c959", None, None, None, [4]]
    assert _is_valid_template(template) is False


def test_valid_template_accepts_16_slots():
    """Pro's live capture has 16 slots; validator must accept >= 15."""
    template = [
        1, None, None, None, "797f3d0293f288ad",
        None, None, 0, [4], None, None,
        3, None, None, 3, 1,
    ]
    assert _is_valid_template(template) is True


def test_valid_template_rejects_non_hex_model_id():
    template = [
        1, None, None, None, "not-sixteen-chr",
        None, None, None, [4], None, None,
        None, None, None, 1,
    ]
    assert _is_valid_template(template) is False


def test_valid_template_rejects_wrong_guard_slot():
    template = [
        1, None, None, None, "fbb127bbb056c959",
        None, None, None, [5], None, None,
        None, None, None, 1,
    ]
    assert _is_valid_template(template) is False


def test_valid_template_rejects_non_int_variant():
    template = [
        1, None, None, None, "fbb127bbb056c959",
        None, None, None, [4], None, None,
        None, None, None, "3",
    ]
    assert _is_valid_template(template) is False


def test_apply_autopatch_returns_zero_without_templates():
    assert apply_autopatch(None) == 0
    assert apply_autopatch({}) == 0
    assert apply_autopatch({"unrelated": "value"}) == 0


def test_apply_autopatch_ignores_malformed_template():
    bad = {"flash": "[not, valid"}
    assert apply_autopatch(bad) == 0


def test_apply_autopatch_ignores_wrong_shape_template():
    bad_shape = {
        "flash": json.dumps([1, None, None, None, "fbb127bbb056c959"])
    }
    assert apply_autopatch(bad_shape) == 0


def test_apply_autopatch_no_change_when_static_already_matches():
    """If the captured raw string equals the current model header, no write."""
    pro_header = Model.G_3_1_PRO.model_header[_JSPB]
    assert apply_autopatch({"pro": pro_header}) == 0


def test_apply_autopatch_does_not_raise_on_none_value():
    """Contract: apply_autopatch never raises, even on unexpected input."""
    assert apply_autopatch({"flash": None}) == 0  # type: ignore[dict-item]


# ---------------------------------------------------------------------------
# R2: Per-model template API (new contract)
#
# These tests pin the rewritten ``apply_autopatch`` signature: it accepts a
# dict keyed by canonical model names ("pro", "flash", "thinking") mapping
# to raw jspb header strings, and writes each string verbatim onto the
# corresponding ``Model.*.model_header`` entry. No cross-model derivation;
# each template is validated independently; missing keys leave the
# corresponding model untouched; 16-slot Pro templates are accepted.
# ---------------------------------------------------------------------------

# Live-capture strings recorded 2026-04-21 (see plan doc). These are the
# exact wire-format payloads Chrome currently sends.
_LIVE_PRO_JSPB = (
    '[1,null,null,null,"797f3d0293f288ad",null,null,0,[4],'
    "null,null,3,null,null,3,1]"
)
_LIVE_FLASH_JSPB = (
    '[1,null,null,null,"56fdd199312815e2",null,null,0,[4],'
    "null,null,3,null,null,1]"
)
# Thinking has no fresh live capture in this session; use a plausible
# 15-slot string shaped like Flash but with Thinking's model_id. The test
# only asserts verbatim round-trip, not semantic correctness of the id.
_PLAUSIBLE_THINKING_JSPB = (
    '[1,null,null,null,"5bf011840784117a",null,null,0,[4],'
    "null,null,3,null,null,1]"
)


def test_apply_autopatch_writes_per_model_templates_verbatim():
    """Pass all three templates; each Model enum gets its own string verbatim."""
    templates = {
        "pro": _LIVE_PRO_JSPB,
        "flash": _LIVE_FLASH_JSPB,
        "thinking": _PLAUSIBLE_THINKING_JSPB,
    }
    patched = apply_autopatch(templates)
    assert patched == 3
    assert Model.G_3_1_PRO.model_header[_JSPB] == _LIVE_PRO_JSPB
    assert Model.G_3_0_FLASH.model_header[_JSPB] == _LIVE_FLASH_JSPB
    assert Model.G_3_0_FLASH_THINKING.model_header[_JSPB] == _PLAUSIBLE_THINKING_JSPB


def test_apply_autopatch_skips_invalid_template_for_one_model_only():
    """Per-model validation: one bad template doesn't veto the others."""
    flash_header_before = Model.G_3_0_FLASH.model_header[_JSPB]
    templates = {
        "pro": _LIVE_PRO_JSPB,
        "flash": "[not, valid json",
    }
    patched = apply_autopatch(templates)
    assert patched == 1
    assert Model.G_3_1_PRO.model_header[_JSPB] == _LIVE_PRO_JSPB
    # Flash left alone because its template failed validation.
    assert Model.G_3_0_FLASH.model_header[_JSPB] == flash_header_before


def test_apply_autopatch_accepts_16_slot_pro_template():
    """Pro's live capture has 16 slots; validator must not reject length > 15."""
    assert len(json.loads(_LIVE_PRO_JSPB)) == 16
    patched = apply_autopatch({"pro": _LIVE_PRO_JSPB})
    assert patched == 1
    assert Model.G_3_1_PRO.model_header[_JSPB] == _LIVE_PRO_JSPB


def test_apply_autopatch_returns_zero_when_none():
    """None input is a no-op: zero patches, all model headers unchanged."""
    snapshot = {
        m: m.model_header[_JSPB]
        for m in (Model.G_3_1_PRO, Model.G_3_0_FLASH, Model.G_3_0_FLASH_THINKING)
    }
    assert apply_autopatch(None) == 0
    for m, header in snapshot.items():
        assert m.model_header[_JSPB] == header


def test_apply_autopatch_does_not_derive_across_models():
    """Passing only 'flash' must NOT mutate Pro or Thinking headers.

    The old API derived all three models from a single capture by swapping
    slot[4] and slot[14]. The new contract forbids that: each model is
    patched iff its own key is present in the dict.
    """
    pro_before = Model.G_3_1_PRO.model_header[_JSPB]
    thinking_before = Model.G_3_0_FLASH_THINKING.model_header[_JSPB]
    patched = apply_autopatch({"flash": _LIVE_FLASH_JSPB})
    assert patched == 1
    assert Model.G_3_0_FLASH.model_header[_JSPB] == _LIVE_FLASH_JSPB
    assert Model.G_3_1_PRO.model_header[_JSPB] == pro_before
    assert Model.G_3_0_FLASH_THINKING.model_header[_JSPB] == thinking_before
