from __future__ import annotations

import json
import os
import time
from pathlib import Path


_DEFAULT_TTL_HOURS = 24.0
_TTL_ENV_VAR = "GEMINI_WAA_CACHE_TTL_HOURS"


def _default_path() -> Path:
    return Path.home() / ".cache" / "gemini_webapi" / "jspb_templates.json"


def _ttl_seconds() -> float:
    raw = os.environ.get(_TTL_ENV_VAR)
    if raw is None:
        return _DEFAULT_TTL_HOURS * 3600.0
    try:
        return float(raw) * 3600.0
    except (TypeError, ValueError):
        return _DEFAULT_TTL_HOURS * 3600.0


def _load_raw(path: Path) -> dict | None:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    return data


def _validate(data: dict) -> bool:
    captured_at = data.get("captured_at")
    source = data.get("source")
    templates = data.get("templates")
    if not isinstance(captured_at, (int, float)):
        return False
    if not isinstance(source, str):
        return False
    if not isinstance(templates, dict):
        return False
    for k, v in templates.items():
        if not isinstance(k, str) or not isinstance(v, str):
            return False
    return True


def read_cache(path: Path | None = None, now: float | None = None) -> dict | None:
    p = path if path is not None else _default_path()
    t = now if now is not None else time.time()
    data = _load_raw(p)
    if data is None or not _validate(data):
        return None
    if t - float(data["captured_at"]) >= _ttl_seconds():
        return None
    return data


def write_cache(
    templates: dict[str, str],
    source: str,
    path: Path | None = None,
    now: float | None = None,
) -> None:
    p = path if path is not None else _default_path()
    t = now if now is not None else time.time()
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {"captured_at": float(t), "source": source, "templates": dict(templates)}
    tmp = p.with_suffix(p.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f)
    os.replace(tmp, p)


def invalidate_cache(
    path: Path | None = None,
    now: float | None = None,
    debounce_hours: float = 1.0,
) -> bool:
    p = path if path is not None else _default_path()
    t = now if now is not None else time.time()
    if not p.exists():
        return False
    data = _load_raw(p)
    if data is not None:
        captured_at = data.get("captured_at")
        if isinstance(captured_at, (int, float)):
            if t - float(captured_at) < debounce_hours * 3600.0:
                return False
    try:
        p.unlink()
    except OSError:
        return False
    return True


def is_cache_fresh(path: Path | None = None, now: float | None = None) -> bool:
    return read_cache(path=path, now=now) is not None
