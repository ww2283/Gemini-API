"""Tests for gemini_webapi.utils.jspb_patch — the auto-capture hotpatch."""
from __future__ import annotations

import json

import pytest

from gemini_webapi.constants import Model
from gemini_webapi.utils import jspb_patch
from gemini_webapi.utils.jspb_patch import (
    _derive_model_header,
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


def test_valid_template_rejects_wrong_length():
    template = [1, None, None, None, "fbb127bbb056c959", None, None, None, [4]]
    assert _is_valid_template(template) is False


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


def test_derive_preserves_target_model_id_and_variant():
    flash_template = [
        1, None, None, None, "fbb127bbb056c959",
        None, None, None, [4], None, None,
        None, None, None, 1,
    ]
    derived = _derive_model_header(flash_template, Model.G_3_1_PRO)
    assert derived is not None
    slots = json.loads(derived)
    assert slots[4] == "797f3d0293f288ad"  # Pro model_id from constants
    assert slots[14] == 3  # Pro variant from constants


def test_derive_wire_format_matches_chrome_no_spaces():
    """Chrome emits jspb headers without whitespace — don't break parity."""
    template = [
        1, None, None, None, "fbb127bbb056c959",
        None, None, None, [4], None, None,
        None, None, None, 1,
    ]
    derived = _derive_model_header(template, Model.G_3_0_FLASH)
    assert derived is not None
    assert " " not in derived


def test_apply_autopatch_returns_zero_without_captured_header():
    assert apply_autopatch(None) == 0
    assert apply_autopatch({}) == 0
    assert apply_autopatch({"unrelated": "value"}) == 0


def test_apply_autopatch_ignores_malformed_template():
    bad = {_JSPB: "[not, valid"}
    assert apply_autopatch(bad) == 0


def test_apply_autopatch_ignores_wrong_shape_template():
    bad_shape = {
        _JSPB: json.dumps([1, None, None, None, "fbb127bbb056c959"])
    }
    assert apply_autopatch(bad_shape) == 0


def test_apply_autopatch_no_change_when_static_already_matches():
    """If library static == captured, patch_count is 0 even on valid input."""
    # Use current static Pro header as "captured"; the derived version for
    # each model should equal the static version, so no writes happen.
    pro_header = Model.G_3_1_PRO.model_header[_JSPB]
    # Can't use Pro's header directly because derive would swap slot 14 to
    # 3 (which matches). But derive also preserves slot 4. So the Pro
    # output matches, but Flash output replaces slot 4+14 accordingly and
    # matches Flash's static. All three should no-op.
    assert apply_autopatch({_JSPB: pro_header}) == 0


def test_apply_autopatch_writes_when_template_has_fresh_structure():
    """Simulate drift: library still has old slot-7=0 format."""
    # Revert Pro header to pre-fix "broken" form
    Model.G_3_1_PRO.model_header[_JSPB] = (
        '[1,null,null,null,"797f3d0293f288ad",null,null,0,[4],null,null,3,null,null,3]'
    )
    Model.G_3_0_FLASH.model_header[_JSPB] = (
        '[1,null,null,null,"fbb127bbb056c959",null,null,0,[4],null,null,1,null,null,1]'
    )
    Model.G_3_0_FLASH_THINKING.model_header[_JSPB] = (
        '[1,null,null,null,"5bf011840784117a",null,null,0,[4],null,null,1,null,null,1]'
    )
    # Capture matches current-Chrome (slot 7 null, slot 11 null)
    captured = (
        '[1,null,null,null,"fbb127bbb056c959",null,null,null,[4],'
        "null,null,null,null,null,1]"
    )
    patched = apply_autopatch({_JSPB: captured})
    assert patched == 3
    # All three models now have the post-drift-fix structure
    for m in (Model.G_3_1_PRO, Model.G_3_0_FLASH, Model.G_3_0_FLASH_THINKING):
        slots = json.loads(m.model_header[_JSPB])
        assert slots[7] is None, f"{m.name} slot 7 should be None"
        assert slots[11] is None, f"{m.name} slot 11 should be None"


def test_apply_autopatch_does_not_raise_on_missing_jspb_patch_dep():
    """Contract: apply_autopatch never raises, even on unexpected input."""
    # Pass garbage to exercise the 'return 0 on any failure' guarantee.
    assert apply_autopatch({_JSPB: None}) == 0  # type: ignore[dict-item]


def test_cache_disabled_by_default(tmp_path, monkeypatch):
    """Without GEMINI_JSPB_AUTOPATCH_CACHE=1, no file should be written."""
    monkeypatch.delenv("GEMINI_JSPB_AUTOPATCH_CACHE", raising=False)
    monkeypatch.setattr(jspb_patch, "_CACHE_PATH", tmp_path / "jspb.json")
    apply_autopatch({
        _JSPB: (
            '[1,null,null,null,"fbb127bbb056c959",null,null,null,[4],'
            'null,null,null,null,null,1]'
        )
    })
    assert not (tmp_path / "jspb.json").exists()


def test_cache_written_when_opted_in(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_JSPB_AUTOPATCH_CACHE", "1")
    monkeypatch.setattr(jspb_patch, "_CACHE_PATH", tmp_path / "jspb.json")
    apply_autopatch({
        _JSPB: (
            '[1,null,null,null,"fbb127bbb056c959",null,null,null,[4],'
            'null,null,null,null,null,1]'
        )
    })
    assert (tmp_path / "jspb.json").exists()
    data = json.loads((tmp_path / "jspb.json").read_text())
    assert isinstance(data["captured_at"], (int, float))
    assert data["template"][4] == "fbb127bbb056c959"
