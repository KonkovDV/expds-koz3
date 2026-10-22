#!/usr/bin/env python3
"""Сборка архива для загрузки на платформу КОЗ №3 и проверки перед отправкой.

    python scripts/build_submission.py --out dist/submission.zip

В архив попадают: metadata.json, inference/ (main.py и lib/*.so), expds/,
configs/, models/ (только файлы, указанные в configs/default.toml, и модели
LightGBM). Проверяется:
* metadata.json заполнен (image не шаблон, entry_point задан);
* нет данных выборок (.las, .laz, .tif, .geojson, .npy, .csv и т. п.) —
  иначе платформа отклонит архив (п. 7.7 ТР), а это ещё и нарушение 13.11;
* размер ≤ 2 ГБ;
* inference/main.py — единственная точка входа в inference/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tomllib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FORBIDDEN = {".las", ".laz", ".tif", ".tiff", ".geojson", ".npy", ".npz", ".csv", ".jpg", ".png", ".jp2",
             ".shp", ".gpkg", ".zip"}
LIMIT = 2 * 1024 ** 3


def collect() -> list[Path]:
    files = [ROOT / "metadata.json", ROOT / "inference" / "main.py"]
    files += sorted((ROOT / "inference" / "lib").glob("expds_core*.so"))
    files += sorted(p for p in (ROOT / "expds").rglob("*.py"))
    files += sorted((ROOT / "configs").glob("*.toml"))
    from expds.pipeline import model_files, model_specs

    cfg = tomllib.loads((ROOT / "configs" / "default.toml").read_text(encoding="utf-8"))
    models = cfg.get("models", {})
    paths = [f for mt in models for p, _ in model_specs(models, mt) for f in model_files(p)]
    for rel in paths + [cfg.get("candidates", {}).get("lgbm_model")]:
        if rel:
            p = ROOT / rel
            if not p.exists():
                sys.exit(f"модель из конфига не найдена: {rel}")
            files.append(p)
    return files


def main() -> None:
    ap = argparse.ArgumentParser(description="Сборка архива решения")
    ap.add_argument("--out", default=str(ROOT / "dist" / "submission.zip"))
    ap.add_argument("--local", action="store_true",
                    help="для проверки распакованного архива в контейнере: без проверки поля image")
    args = ap.parse_args()
    meta = json.loads((ROOT / "metadata.json").read_text(encoding="utf-8"))
    problems = []
    if not args.local and ("ВАШ_" in meta.get("image", "") or not meta.get("image")):
        problems.append("metadata.json: укажите публичный образ Docker Hub в поле image")
    if not meta.get("entry_point"):
        problems.append("metadata.json: нет entry_point")
    files = collect()
    for f in files:
        if f.suffix.lower() in FORBIDDEN:
            problems.append(f"запрещённый тип файла в архиве: {f.relative_to(ROOT)}")
    extra_py = [p for p in (ROOT / "inference").glob("*.py") if p.name != "main.py"]
    if extra_py:
        problems.append("в inference/ должна быть одна точка входа, лишние: " + ", ".join(p.name for p in extra_py))
    if not list((ROOT / "inference" / "lib").glob("expds_core*.so")):
        print("предупреждение: в inference/lib нет expds_core*.so — на платформе будет NumPy (медленнее)")
    if problems:
        sys.exit("\n".join(problems))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f, f.relative_to(ROOT).as_posix())
    size = out.stat().st_size
    if size > LIMIT:
        sys.exit(f"архив {size / 2**20:.0f} МБ больше 2 ГБ")
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    print(f"архив: {out} — {len(files)} файлов, {size / 2**20:.1f} МБ, sha256 {digest}")
    for f in files:
        print("  ", f.relative_to(ROOT).as_posix())


if __name__ == "__main__":
    main()
