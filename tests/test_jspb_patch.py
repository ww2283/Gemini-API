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


def test_valid_template_accepts_short_template_with_intact_guard():
    """Short templates aren't garbage if slot 0 is the protocol sentinel and
    the guard at slot 8 is intact. The validator no longer enforces a
    minimum length of 15 -- the harvester only sees wire-valid templates,
    so capture-error detection is enough.
    """
    template = [1, None, None, None, None, None, None, None, [4]]
    assert _is_valid_template(template) is True


def test_valid_template_accepts_16_slots():
    """Pro's live capture has 16 slots; validator must accept >= 15."""
    template = [
        1, None, None, None, "797f3d0293f288ad",
        None, None, 0, [4], None, None,
        3, None, None, 3, 1,
    ]
    assert _is_valid_template(template) is True


def test_valid_template_accepts_string_at_slot_14_variant():
    """The validator no longer polices slot-14 type. A wire-captured
    template with a string at slot 14 is accepted; only protocol-level
    structure (slot 0 sentinel + slot 8 guard) is policed.
    """
    template = [
        1, None, None, None, "fbb127bbb056c959",
        None, None, None, [4], None, None,
        None, None, None, "3",
    ]
    assert _is_valid_template(template) is True


def test_valid_template_accepts_null_model_id_at_slot_4():
    """Slot-4 leniency is structural, not specific to the live 17-slot Pro
    shape. A minimal 15-slot template with only slot 4 mutated to null is
    accepted by the permissive validator.
    """
    template = [
        1, None, None, None, None,
        None, None, None, [4], None, None,
        None, None, None, 1,
    ]
    assert _is_valid_template(template) is True


def test_valid_template_rejects_wrong_guard_slot():
    template = [
        1, None, None, None, "fbb127bbb056c959",
        None, None, None, [5], None, None,
        None, None, None, 1,
    ]
    assert _is_valid_template(template) is False


def test_valid_template_rejects_empty_list():
    """An empty list isn't a template -- it's a capture error."""
    assert _is_valid_template([]) is False


def test_valid_template_rejects_when_slot_0_not_protocol_sentinel():
    """Slot 0 is the one immutable schema marker. A structurally
    well-formed template with a non-1 slot 0 is a protocol-level shift
    and must be rejected.
    """
    template = [
        2, None, None, None, "fbb127bbb056c959",
        None, None, None, [4], None, None,
        None, None, None, 1,
    ]
    assert _is_valid_template(template) is False


def test_apply_autopatch_returns_zero_without_templates():
    assert apply_autopatch(None) == 0
    assert apply_autopatch({}) == 0
    assert apply_autopatch({"unrelated": "value"}) == 0


def test_apply_autopatch_ignores_malformed_template():
    bad = {"flash": "[not, valid"}
    assert apply_autopatch(bad) == 0


def test_apply_autopatch_engages_on_short_intact_template():
    """Validator accepts -> hotpatch engages, even on short shapes.

    Pins the new permissive contract end-to-end: a 9-slot template with
    slot 0 == 1 and slot 8 == [4] is accepted by the validator, so
    apply_autopatch writes it verbatim to Flash's model_header.
    """
    raw = json.dumps(
        [1, None, None, None, "fbb127bbb056c959", None, None, None, [4]]
    )
    patched = apply_autopatch({"flash": raw})
    assert patched == 1
    assert Model.G_3_0_FLASH.model_header[_JSPB] == raw


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


# ---------------------------------------------------------------------------
# Bugfix slate 2026-04-27 -- R1: permissive validator for slot-4=null
#
# Live Chrome capture at 12:00 EDT 2026-04-27 shows Google has dropped the
# 16-char hex model_id from slot 4 -- the wire-format payload now sends
# ``null`` there. The current strict validator rejects this, so
# ``apply_autopatch`` silently skips Pro, the runtime falls back to the
# stale static ``Model.G_3_1_PRO.model_header``, and StreamGenerate returns
# status [5]. Real-world impact: Valuator_AI's deep-reasoning fell back from
# webapi to CLI at 2026-04-27T16:24Z. The two tests below pin the bug as
# failing assertions; the forward-looking permissive contract (length
# variations, slot-14 leniency, empty-list rejection, slot-0 sentinel) is
# pinned in R2.
# ---------------------------------------------------------------------------

# Exact 17-slot live-capture string from
# ``~/.cache/gemini_webapi/jspb_templates.json`` (captured 2026-04-27 12:00
# EDT). Slot 4 is null (model_id dropped); slot 8 is [4] (conv-header
# guard); slot 14 is 3 (variant). Trailing slot is a UUID.
_BUG_LIVE_PRO_JSPB_SLOT4_NULL = (
    '[1,null,null,null,null,null,null,0,[4],null,null,3,null,null,3,null,'
    '"DB946E7A-9AB9-48B4-8F74-BDBE853818A5"]'
)


def test_valid_template_accepts_null_at_slot_4():
    """Live Pro capture has null at slot 4; validator must accept it.

    Google has dropped the 16-char hex model_id field. The validator
    currently rejects this payload because slot 4 is not a string -- this
    causes apply_autopatch to skip Pro and the runtime to fall back to the
    drifted static header.
    """
    template = json.loads(_BUG_LIVE_PRO_JSPB_SLOT4_NULL)
    assert _is_valid_template(template) is True


def test_apply_autopatch_engages_for_slot_4_null_template():
    """End-to-end: validator accepts -> hotpatch writes Pro header verbatim.

    Pins the engagement of the entire autopatch chain for the new wire
    shape. Today this fails because the validator rejects null-at-slot-4,
    so apply_autopatch returns 0 and Model.G_3_1_PRO.model_header is left
    pointing at the stale static template.
    """
    patched = apply_autopatch({"pro": _BUG_LIVE_PRO_JSPB_SLOT4_NULL})
    assert patched == 1
    assert (
        Model.G_3_1_PRO.model_header[_JSPB]
        == _BUG_LIVE_PRO_JSPB_SLOT4_NULL
    )


# ---------------------------------------------------------------------------
# R2: static Pro template refresh (cold-boot regression guard)
#
# Cold-boot users (no harvester, no cache) only have the static
# ``Model.G_3_1_PRO.model_header`` to fall back on. Today that static is
# the drifted Aug-2025-era shape (16 slots, slot 4 = "797f3d0293f288ad",
# slot 7 = None, slot 11 = None, slot 14 = 3) which the server rejects
# with status [5]. G2 refreshes the static to match the live wire shape;
# the tests below pin that refresh and guard against G2 over-reaching
# into Flash/Thinking.
# ---------------------------------------------------------------------------

# Captured 2026-04-27 from Valuator_AI managed-profile harvester. Slot 16 is a
# per-session UUID; the static-refresh test pins shape (length, slots 4/7/8/11/14)
# but not the volatile UUID slot.
_STATIC_REFRESH_PRO_SHAPE_LIVE = (
    '[1,null,null,null,null,null,null,0,[4],'
    'null,null,3,null,null,3,null,null]'
)


def test_static_pro_template_passes_permissive_validator():
    """Regression guard: the static Pro template is wire-valid under the
    permissive validator. Passes today (validator is permissive enough to
    accept the still-drifted static); future static edits that smuggle in
    an invalid slot-0 / slot-8 value will be caught here.
    """
    parsed = _parse_jspb(Model.G_3_1_PRO.model_header[_JSPB])
    assert parsed is not None
    assert _is_valid_template(parsed) is True


def test_static_pro_template_has_refreshed_shape():
    """The static Pro template must match the live 17-slot wire shape.

    Cold-boot users (no harvester, no cache) fall back to this static.
    The drifted Aug-2025 static (len 15, slot 4 = "797f3d0293f288ad",
    slot 7 = None, slot 11 = None) triggers status [5] from the server.
    G2 refreshes the static to match ``_STATIC_REFRESH_PRO_SHAPE_LIVE``
    (slot 16 set to null since the volatile per-session UUID can't be
    pinned in source). This test fails until that refresh lands.
    """
    raw = Model.G_3_1_PRO.model_header[_JSPB]
    assert raw == _STATIC_REFRESH_PRO_SHAPE_LIVE
    parsed = _parse_jspb(raw)
    assert parsed is not None
    assert len(parsed) >= 17
    assert parsed[0] == 1
    assert parsed[4] is None
    assert parsed[7] == 0
    assert parsed[8] == [4]
    assert parsed[11] == 3
    assert parsed[14] == 3


def test_static_flash_and_thinking_templates_unchanged_by_pro_refresh():
    """G2 only refreshes Pro. Flash and Thinking statics stay at their
    canonical Flash-family shape (16-char hex at slot 4, variant 1 at
    slot 14). Guards against G2 over-reaching.
    """
    flash_parsed = _parse_jspb(Model.G_3_0_FLASH.model_header[_JSPB])
    thinking_parsed = _parse_jspb(Model.G_3_0_FLASH_THINKING.model_header[_JSPB])
    assert flash_parsed is not None
    assert thinking_parsed is not None
    assert isinstance(flash_parsed[4], str) and len(flash_parsed[4]) == 16
    assert isinstance(thinking_parsed[4], str) and len(thinking_parsed[4]) == 16
    assert flash_parsed[14] == 1
    assert thinking_parsed[14] == 1
