"""Проверка сборки кэша HF из «сплющенного» бандла (без загрузки весов)."""
import os, json, tarfile, shutil, sys
from pathlib import Path
src = (Path(__file__).resolve().parents[2] / "notebooks/k01/c2_code.py").read_text(encoding="utf-8")
src = src[:src.index("SMP_FROM = setup_smp()")]
T = Path("/tmp/bot9_hf"); shutil.rmtree(T, ignore_errors=True)
bundle = T / "input/koz3-li-tin-59-t128-bundle"; hf = bundle / "hf_hub_cache"
g = {"CODE_DIR": Path("/nonexistent"), "WORK": T / "work", "CFG": {"pretrained": True, "models": [{"encoder": "resnet18"}, {"encoder": "resnet34"}]},
     "os": os, "tarfile": tarfile, "shutil": shutil, "sys": sys, "Path": Path, "_ls": lambda *a: []}
exec(src.replace('Path("/kaggle/input")', 'Path("%s")' % (T / "input")), g)
for enc, (repo, snap, c, w) in g["HF_SPECS"].items():
    (hf / c).parent.mkdir(parents=True, exist_ok=True); (hf / c).write_text('{"a":1}')
    (hf / w).parent.mkdir(parents=True, exist_ok=True); (hf / w).write_bytes(b"0" * 2_000_000)
(bundle / "expds_code.tar.gz").write_bytes(b"")
print("code_dir:", g["resolve_code_dir"]())
print(json.dumps(g["setup_hf_cache"](), ensure_ascii=False))
for p in sorted((T / "work/hf_hub_cache").rglob("*")):
    print(p.relative_to(T / "work"), "->", os.readlink(p) if p.is_symlink() else "")
os.unlink(hf / g["HF_SPECS"]["resnet34"][3]); shutil.rmtree(T / "work")
try:
    g["setup_hf_cache"](); print("НЕ УПАЛО")
except FileNotFoundError as e:
    print("ожидаемая ошибка:", str(e)[:120])
