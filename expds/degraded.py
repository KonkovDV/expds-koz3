"""W4 (01.11): маркер деградации v3. Ветка v3 (D-FINE, шлюз городищ, frame_prior) не загрузилась или упала →
громкий WARNING «V3_DEGRADED» в лог и stderr + файл V3_DEGRADED.json рядом с файлом ответа (причины списком).
check_submission считает и строку, и файл FAIL. Выдача при этом не прерывается (как раньше — как v2)."""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

log = logging.getLogger(__name__)
MARKER = "V3_DEGRADED.json"
TAG = "V3_DEGRADED"
_OUT: Path | None = None
_REASONS: list[dict] = []


def set_output(output_path) -> None:
    """В начале прогона: каталог ответа; старый маркер прошлого прогона в нём удаляется."""
    global _OUT
    _OUT = Path(output_path).resolve().parent / MARKER
    _REASONS.clear()
    try:
        _OUT.unlink(missing_ok=True)
    except OSError:
        log.exception("не удалён старый %s", _OUT)


def mark(component: str, reason: str) -> None:
    item = {"component": component, "reason": str(reason)[:1000], "t": time.strftime("%Y-%m-%dT%H:%M:%S")}
    _REASONS.append(item)
    msg = f"!!! {TAG}: {component}: {item['reason']} — ветка v3 не работает, выдача как v2 !!!"
    log.warning(msg)
    try:
        print(msg, file=sys.stderr, flush=True)
    except Exception:  # noqa: BLE001
        pass
    if _OUT is not None:
        try:
            _OUT.parent.mkdir(parents=True, exist_ok=True)
            _OUT.write_text(json.dumps({"degraded": True, "reasons": _REASONS}, ensure_ascii=False, indent=1),
                            encoding="utf-8")
        except OSError:
            log.exception("%s: маркер %s не записан", TAG, _OUT)


def reasons() -> list[dict]:
    return list(_REASONS)
