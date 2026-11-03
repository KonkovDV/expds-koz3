"""Регрессии аудита 03.11 (docs/AUDIT_2026-11-03.md): путь сдачи пишет файл ответа и под сигналом."""
from __future__ import annotations

import json
import signal
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_sigterm_still_writes_when_postprocessing_fails(tmp_path, monkeypatch):
    """A1: обработчик SIGTERM шёл через finish(); исключение в постобработке оставляло площадку без файла."""
    from shapely.geometry import Point

    from expds import watchdog

    def boom(feats, cfg):
        raise RuntimeError("постобработка упала")

    monkeypatch.setattr(watchdog, "finish", boom)
    out = tmp_path / "result.geojson"
    state = watchdog.RunState(out, {})
    state.add([{"geom": Point(4_000_000.0, 7_000_000.0).buffer(3.0), "class_name": "kurgany_tselye",
                "confidence": 0.9, "region_name": "r1", "markup_type": "li"}])
    exits: list[int] = []
    previous = watchdog.arm_signal(state, exit_fn=exits.append)
    try:
        signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
    finally:
        signal.signal(signal.SIGTERM, previous)
    assert exits == [0]
    fc = json.loads(out.read_text(encoding="utf-8"))
    assert [f["properties"]["class_name"] for f in fc["features"]] == ["kurgany_tselye"]


def test_sigterm_handler_stays_armed_through_crop_verify(tmp_path, monkeypatch):
    """A2: run() снимал обработчик SIGTERM до проверки кропа и итоговой записи; сигнал там убивал процесс без файла."""
    from expds import crop_model, pipeline

    cfg = tomllib.loads((ROOT / "configs" / "default.toml").read_text(encoding="utf-8"))
    cfg["time"]["watchdog"] = False
    cfg["verify"]["enabled"] = True
    cfg_path = tmp_path / "cfg.toml"
    cfg_path.write_text("", encoding="utf-8")
    monkeypatch.setattr(pipeline, "load_config", lambda _p: cfg)
    seen: list = []

    def fake_verify(feats, regions, vcfg):
        seen.append(signal.getsignal(signal.SIGTERM))
        return None

    monkeypatch.setattr(crop_model, "maybe_verify", fake_verify)
    before = signal.getsignal(signal.SIGTERM)
    (tmp_path / "input").mkdir()
    pipeline.run(tmp_path / "input", tmp_path / "out" / "result.geojson", cfg_path, ROOT)
    assert seen and seen[0] is not before            # во время проверки кропа обработчик run() ещё стоит
    assert signal.getsignal(signal.SIGTERM) is before   # после итоговой записи прежний обработчик вернулся
    assert (tmp_path / "out" / "result.geojson").is_file()


def test_sp_det2_stays_out_of_the_archive_and_the_run_when_off():
    """Интеграция 03.11: sat2 (bots/combo-sat2-fr1h) подвязан за флагом [sp_det2]; по умолчанию не входит никуда."""
    cfg = tomllib.loads((ROOT / "configs" / "default.toml").read_text(encoding="utf-8"))
    assert not (cfg.get("sp_det2") or {}).get("enabled")
    src = (ROOT / "scripts" / "build_submission.py").read_text(encoding="utf-8")
    assert '"sp_det2")' in src.replace("\n", "").replace(" ", "")[:20000] or '"sp_det2"' in src
    from expds import pipeline

    assert callable(pipeline._sp_det2_phase)


def test_no_en_lineage_module_in_tree():
    """Интеграция 03.11: ветки с expds/geo_pixgbm.py влиты без него (правило DOC-1)."""
    assert not (ROOT / "expds" / "geo_pixgbm.py").exists()
