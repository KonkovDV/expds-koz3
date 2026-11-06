""" (29.10, ПК e2e в pantum/n7q:20261024): «SP_DFINE_UNAVAILABLE … vendor: ValueError: Unable to compare
versions for tqdm>=4.60: need=4.60 found=None». У tqdm в образе dist-info без версии → importlib.metadata.version()
= None → transformers.dependency_versions_check падает при import transformers.

Защита (expds/sp_dfine.py): (a) tqdm вендорен с dist-info; (b) шим версии для пакетов образа (module.__version__ →
версия из имени dist-info) и пропуск require_version, если версия не определяется, а модуль импортируется — с громким
SP_DFINE_DEPMETA. Настоящие отказы (модуля нет, версия слишком старая) по-прежнему → SP_DFINE_UNAVAILABLE.
Всё — в отдельных чистых процессах (метаданные pytest-процесса не трогаем)."""
from pathlib import Path

import pytest

from test_bot7_dfine_import import VENDOR, _probe, _root, _run

pytestmark = pytest.mark.skipif(not (VENDOR / "transformers" / "__init__.py").is_file(),
                                reason=f"нет вендора transformers: {VENDOR}")


def _vendor_without_tqdm(tmp: Path) -> Path:
    v = tmp / "vendor_no_tqdm"
    v.mkdir()
    for p in VENDOR.iterdir():
        if not p.name.lower().startswith("tqdm"):
            (v / p.name).symlink_to(p, target_is_directory=p.is_dir())
    return v


def _shadow(tmp: Path, *dist_dirs: str, metadata: dict | None = None) -> Path:
    s = tmp / "shadow"
    for d in dist_dirs:
        (s / d).mkdir(parents=True)
        if metadata and d in metadata:
            (s / d / "METADATA").write_text(metadata[d], encoding="utf-8")
    return s


NOMETA = ("tqdm-4.70.1.dist-info", "packaging-26.3.dist-info", "filelock-4.2.4.dist-info", "numpy-2.5.4.dist-info",
          "huggingface_hub-2.2.0.dist-info", "safetensors-0.8.0.dist-info", "PyYAML-6.0.3.dist-info",
          "torch-2.11.0.dist-info", "torchvision-0.26.0.dist-info")

NOSHIM = "sp_dfine._metadata_shim = lambda: None"

# tqdm «отсутствует» вне вендора: любая находка tqdm вне VENDOR_DIR → ModuleNotFoundError
BLOCK_TQDM = """
import importlib.abc, importlib.machinery
class _NoTqdm(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name != "tqdm" and not name.startswith("tqdm."):
            return None
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        if spec is None or not str(spec.origin or "").startswith(VENDOR_DIR):
            raise ModuleNotFoundError(f"No module named {name!r} (имитация: нет tqdm в образе)", name=name)
        return spec
sys.meta_path.insert(0, _NoTqdm())
"""


def test_repro_hidden_tqdm_metadata_without_fix_fails(tmp_path):
    # воспроизведение ошибки: tqdm не вендорен, шим выключен
    out = _probe(_root(tmp_path, _vendor_without_tqdm(tmp_path)), _shadow(tmp_path, "tqdm-4.70.1.dist-info"),
                 pre="from expds import sp_dfine\n" + NOSHIM)
    assert not out["ok"]
    assert "SP_DFINE_UNAVAILABLE" in out["err"] and "Unable to compare versions for tqdm" in out["err"], out


def test_hidden_tqdm_metadata_shim_alone(tmp_path):
    # без вендорного tqdm: шим берёт версию из имени dist-info
    out = _probe(_root(tmp_path, _vendor_without_tqdm(tmp_path)), _shadow(tmp_path, "tqdm-4.70.1.dist-info"))
    assert out["ok"] and out["src"] == "vendor" and out["ver"] == "5.21.0", out


def test_hidden_tqdm_metadata_nameless_dist_info(tmp_path):
    # версию не определить ни из METADATA, ни из имени: require_version пропускается (модуль импортируется)
    out = _probe(_root(tmp_path, _vendor_without_tqdm(tmp_path)), _shadow(tmp_path, "tqdm.dist-info"))
    assert out["ok"] and out["src"] == "vendor", out


def test_hidden_tqdm_metadata_vendored_tqdm(tmp_path):
    if not (VENDOR / "tqdm" / "__init__.py").is_file():
        pytest.skip("в вендоре нет tqdm")
    out = _probe(_root(tmp_path, VENDOR), _shadow(tmp_path, "tqdm-4.70.1.dist-info"), pre="from expds import sp_dfine\n" + NOSHIM)
    # даже без шима: метаданные вендорного tqdm раньше на sys.path
    assert out["ok"] and out["src"] == "vendor", out


def test_all_image_deps_metadata_hidden(tmp_path):
    code = f"""
        ROOTP = {str(_root(tmp_path, VENDOR))!r}
        from pathlib import Path
        from expds import sp_dfine
        try:
            C, M, info = sp_dfine.import_dfine(Path(ROOTP))
            print("J=" + json.dumps({{"ok": True, "src": info["source"], "fb": info.get("meta_fallback")}}))
        except ImportError as e:
            print("J=" + json.dumps({{"ok": False, "err": str(e)}}))
    """
    out = _run(code, _shadow(tmp_path, *NOMETA))
    assert out["ok"] and out["src"] == "vendor", out
    assert {"numpy", "torch", "pyyaml", "huggingface-hub"} <= set(out["fb"]), out


def test_tqdm_absent_without_vendored_tqdm_fails_loudly(tmp_path):
    v = _vendor_without_tqdm(tmp_path)
    out = _probe(_root(tmp_path, v), pre=f"VENDOR_DIR = {str(tmp_path / 'pkg')!r}\n" + BLOCK_TQDM)
    assert not out["ok"] and "SP_DFINE_UNAVAILABLE" in out["err"] and "tqdm" in out["err"], out


def test_tqdm_absent_with_vendored_tqdm_succeeds(tmp_path):
    if not (VENDOR / "tqdm" / "__init__.py").is_file():
        pytest.skip("в вендоре нет tqdm")
    root = _root(tmp_path, VENDOR)
    code = (f"VENDOR_DIR = ({str(root)!r}, {str(VENDOR.resolve())!r})\n" + BLOCK_TQDM + f"""
from pathlib import Path
from expds import sp_dfine
try:
    C, M, info = sp_dfine.import_dfine(Path({str(root)!r}))
    import tqdm
    print("J=" + json.dumps({{"ok": True, "src": info["source"], "tqdm": tqdm.__file__, "v": tqdm.__version__}}))
except ImportError as e:
    print("J=" + json.dumps({{"ok": False, "err": str(e)}}))
""")
    out = _run(code)
    assert out["ok"] and out["src"] == "vendor", out
    assert out["tqdm"].startswith((str(root), str(VENDOR.resolve()))) and out["v"] == "4.70.1", out


def test_too_old_dependency_still_fails_loudly(tmp_path):
    # шим не глушит настоящий конфликт версий: numpy 1.0 в метаданных → ImportError → SP_DFINE_UNAVAILABLE
    sh = _shadow(tmp_path, "numpy-1.0.dist-info",
                 metadata={"numpy-1.0.dist-info": "Metadata-Version: 2.1\nName: numpy\nVersion: 1.0\n"})
    out = _probe(_root(tmp_path, VENDOR), sh)
    assert not out["ok"] and "SP_DFINE_UNAVAILABLE" in out["err"] and "numpy" in out["err"], out


def test_shim_leaves_unknown_packages_alone(tmp_path):
    out = _run("""
        import importlib.metadata as md
        from expds import sp_dfine
        sp_dfine._metadata_shim(); sp_dfine._metadata_shim()
        try:
            md.version("b7-no-such-package")
            r = "found"
        except md.PackageNotFoundError:
            r = "notfound"
        print("J=" + json.dumps({"r": r, "np": md.version("numpy"), "shim": getattr(md.version, "_b7_shim", False)}))
    """)
    assert out["r"] == "notfound" and out["np"][0].isdigit() and out["shim"], out
