"""[verify] weights от корня архива, не от cwd. Тест перенесён из tests/test_bot8_faults.py
без остальных тестов 9316b5f (в кандидат взята только правка кропа)."""


def test_verify_weights_resolve_from_archive_root_not_cwd(tmp_path, monkeypatch):
    """crop_model.py:32-34 — Path(weights) от текущей папки; запуск не из корня молча выключал проверку."""
    from conftest import ROOT

    from expds.pipeline import rooted_verify

    monkeypatch.chdir(tmp_path)
    cfg = {"verify": {"enabled": True, "weights": "models/sp_crop.pt"}, "x": 1}
    got = rooted_verify(cfg, ROOT)
    assert got["verify"]["weights"] == str(ROOT / "models/sp_crop.pt") and got["x"] == 1
    assert cfg["verify"]["weights"] == "models/sp_crop.pt"
