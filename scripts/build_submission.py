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
import re
import sys
import tomllib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FORBIDDEN = {".las", ".laz", ".tif", ".tiff", ".geojson", ".npy", ".npz", ".csv", ".jpg", ".png", ".jp2",
             ".shp", ".gpkg", ".zip"}
LIMIT = 2 * 1024 ** 3
#: bot-7 (v3, red team bot-4 В6): модули обучения/разбора, недостижимые из inference/main.py (AST-граф
#: импортов, bot-7/work6/import_graph.py). В репозитории остаются (скрипты и тесты), в архив не идут.
TRAIN_ONLY = {"gt.py", "blobs.py", "coarse.py", "det_boxes.py"}
FLAG_MODULES = ("sp_dfine", "gor_gate", "dfine_aeor", "orgate")   # bot-7 v3: секция конфига = имя модуля expds/<имя>.py


def collect() -> list[Path]:
    files = [ROOT / "metadata.json", ROOT / "inference" / "main.py"]
    files += sorted((ROOT / "inference" / "lib").glob("expds_core*.so"))
    cfg = tomllib.loads((ROOT / "configs" / "default.toml").read_text(encoding="utf-8"))
    # bot-7 v3: модули за флагом (ленивый импорт в pipeline.py только при enabled) кладутся, только если флаг включён:
    # выключенный sp_dfine не тянет transformers в граф импортов архива (check_submission imports)
    off = {f"{sec}.py" for sec in FLAG_MODULES if not (cfg.get(sec) or {}).get("enabled")}
    if (cfg.get("gor_gate") or {}).get("enabled") or (cfg.get("dfine_aeor") or {}).get("enabled"):
        off.discard("sp_dfine.py")   # gor_gate берёт закреплённый импорт transformers из sp_dfine.import_dfine
    files += sorted(p for p in (ROOT / "expds").rglob("*.py")
                    if not (p.parent == ROOT / "expds" and (p.name in TRAIN_ONLY or p.name in off)))
    files += sorted((ROOT / "configs").glob("*.toml"))
    from expds.pipeline import model_files, model_specs

    models = cfg.get("models", {})
    paths = [f for mt in models for p, _ in model_specs(models, mt) for f in model_files(p)]
    verify = (cfg.get("verify") or {}).get("weights") if (cfg.get("verify") or {}).get("enabled") else None
    seen = set()
    rr = cfg.get("rerank_arch") or {}
    rerank = [rr.get("model", "models/rerank_arch_lgbm.txt"), rr.get("meta", "models/rerank_arch_lgbm.json")] if rr.get("enabled") else []   # bot-7: только если флаг включён
    og = cfg.get("orgate") or {}
    # bot-6: гейт ORGATE G (модель + meta) — только если [orgate] enabled
    rerank += [og.get("model", "models/orgate_g.txt"), og.get("meta", "models/orgate_g.json")] if og.get("enabled") else []
    pix = (cfg.get("geophys") or {}).get("pixgbm") or {}
    # bot-7: модели PIXGBM (poly.txt, finds.txt) — только если [geophys.pixgbm] enabled
    pixm = [f"{pix.get('model_dir', 'models/pixgbm/all').rstrip('/')}/{k}.txt" for k in ("poly", "finds")] \
        if pix.get("enabled") else []
    sd = cfg.get("sp_dfine") or {}
    # bot-7 v3: D-FINE (каталоги config.json + model.safetensors) и вендор колёс — только если [sp_dfine] enabled
    dfm = [f"{str(w).rstrip('/')}/{f}" for w in sd.get("weights", []) for f in ("config.json", "model.safetensors")] \
        if sd.get("enabled") else []
    gg = cfg.get("gor_gate") or {}
    # bot-7 v3: шлюз городищ bot-5 (DINOv2-S config.json + model.safetensors, голова npz) — только если [gor_gate] enabled
    if gg.get("enabled"):
        bb = str(gg.get("backbone", "models/dinov2-small")).rstrip("/")
        dfm += [f"{bb}/config.json", f"{bb}/model.safetensors", gg.get("head", "models/gor_gate/gor_v3_head.npz")]
    ae = cfg.get("dfine_aeor") or {}
    # bot-6: D-FINE Ae/Or — каталоги весов и файл порогов OOF только если [dfine_aeor] enabled
    if ae.get("enabled"):
        w = ae.get("weights") or {}
        wl = list(w) if isinstance(w, list) else [x for v in w.values() for x in ([v] if isinstance(v, str) else v)]
        absw = [x for x in wl if Path(str(x)).is_absolute()]
        if absw:
            sys.exit(f"[dfine_aeor] weights: абсолютные пути {absw} в архив не пакуются — свяжите их в models/ "
                     f"(ln -s <каталог bot-9> models/dfine_ae) и укажите относительный путь")
        dfm += [f"{str(x).rstrip('/')}/{f}" for x in wl for f in ("config.json", "model.safetensors")]
        if ae.get("thresholds_file"):
            dfm.append(ae["thresholds_file"])
    if sd.get("enabled") or gg.get("enabled") or ae.get("enabled"):
        vend = ROOT / "inference" / "lib" / "vendor"
        if not vend.is_dir():
            sys.exit("[sp_dfine]/[gor_gate] enabled, а inference/lib/vendor нет (распакуйте колёса bot-9 submission_dfine/wheels)")
        files += sorted(q for q in vend.rglob("*") if q.is_file() and "__pycache__" not in q.parts)
    for rel in paths + [cfg.get("candidates", {}).get("lgbm_model"), verify] + rerank + pixm + dfm:
        if not rel or rel in seen:
            continue
        seen.add(rel)
        p = ROOT / rel
        if not p.exists():
            sys.exit(f"модель из конфига не найдена: {rel}")
        files.append(p)
    return files


def toml_with_only_classes(text: str, classes: list[str]) -> str:
    """Дописывает [post] only_classes в текст конфига. Файл репозитория не трогает.

    Ключ нужен переписи одного класса. Без него архив совпадает с обычной сборкой.
    """
    quoted = ", ".join(json.dumps(name, ensure_ascii=False) for name in classes)
    return text.rstrip() + f"\n\n[post]\nonly_classes = [{quoted}]\n"


_HEADER = re.compile(r"^\s*\[([^\]]+)\]\s*(#.*)?$")
_KEY = re.compile(r"^(\s*)([A-Za-z_]+)(\s*=\s*)([0-9.eE+-]+)(.*)$")


def toml_with_thresholds(text: str, changes: dict[str, dict[str, tuple[float, float]]]) -> str:
    """Меняет значения в `[thresholds_by_markup.<группа>]`: {группа: {класс: (было, стало)}}.

    Строка правится на месте, комментарии остаются. Значение в файле должно совпасть с «было»:
    иначе вариант посчитан не от этого конфига.
    """
    table, done, out = None, set(), []
    for line in text.splitlines(keepends=True):
        head = _HEADER.match(line)
        if head:
            table = head.group(1).strip()
        key = _KEY.match(line.rstrip("\r\n"))
        group = table.split(".", 1)[1] if table and table.startswith("thresholds_by_markup.") else None
        if key and group in changes and key.group(2) in changes[group]:
            old, new = changes[group][key.group(2)]
            if abs(float(key.group(4)) - old) > 1e-9:
                sys.exit(f"[{table}] {key.group(2)} = {key.group(4)}, а вариант считали от {old}")
            end = line[len(line.rstrip("\r\n")):]
            line = f"{key.group(1)}{key.group(2)}{key.group(3)}{round(new, 6)!r}{key.group(5)}{end}"
            done.add((group, key.group(2)))
        out.append(line)
    missing = {(g, c) for g, cls in changes.items() for c in cls} - done
    if missing:
        sys.exit(f"в конфиге нет порогов {sorted(missing)}")
    return "".join(out)


def variant_from_zip(src: Path, out: Path, classes: list[str] | None = None,
                     thresholds: dict[str, dict[str, tuple[float, float]]] | None = None) -> str:
    """Перепись или вариант порогов из уже оценённого архива: все файлы те же байты, меняется только конфиг.

    Дерево репозитория после загрузки уходит вперёд, а замер должен мерить тот самый архив.
    """
    name = "configs/default.toml"
    with zipfile.ZipFile(src) as zin:
        text = zin.read(name).decode("utf-8")
        config_text = text
        if thresholds:
            config_text = toml_with_thresholds(config_text, thresholds)
        if classes:
            if "post" in tomllib.loads(text):
                sys.exit(f"{src}: в конфиге уже есть [post], ключ переписи не дописать")
            config_text = toml_with_only_classes(config_text, classes)
        cfg = tomllib.loads(config_text)
        for group, cls in (thresholds or {}).items():
            for c, (_old, new) in cls.items():
                if abs(cfg["thresholds_by_markup"][group][c] - round(new, 6)) > 1e-9:
                    sys.exit(f"[thresholds_by_markup.{group}] {c}: после правки не {new}")
        out.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zout:
            for info in zin.infolist():
                if info.filename == name:
                    zout.writestr(name, config_text)
                else:
                    zout.writestr(info, zin.read(info.filename), compress_type=zipfile.ZIP_DEFLATED)
    return hashlib.sha256(out.read_bytes()).hexdigest()


def census_from_zip(src: Path, out: Path, classes: list[str]) -> str:
    return variant_from_zip(src, out, classes=classes)


def density_changes(path: Path) -> dict[str, dict[str, tuple[float, float]]]:
    """Пороги из отчёта `scripts/density_thresholds.py`: только те, что сдвинулись."""
    variant = json.loads(Path(path).read_text(encoding="utf-8"))["variant"]
    return {g: {c: (r["tau"], r["tau_new"]) for c, r in cls.items() if r["tau_new"] != r["tau"]}
            for g, cls in variant.items() if any(r["tau_new"] != r["tau"] for r in cls.values())}


def main() -> None:
    ap = argparse.ArgumentParser(description="Сборка архива решения")
    ap.add_argument("--out", default=str(ROOT / "dist" / "submission.zip"))
    ap.add_argument("--local", action="store_true",
                    help="для проверки распакованного архива в контейнере: без проверки поля image")
    ap.add_argument("--only-class", action="append", default=[],
                    help="перепись: в архивный конфиг дописать [post] only_classes, репозиторий не менять")
    ap.add_argument("--from-zip", default=None,
                    help="перепись или вариант порогов из уже загруженного архива: код и веса оттуда, меняется только конфиг")
    ap.add_argument("--thresholds", default=None,
                    help="с --from-zip: пороги из отчёта scripts/density_thresholds.py (вариант раздела 5)")
    args = ap.parse_args()
    if args.from_zip:
        changes = density_changes(Path(args.thresholds)) if args.thresholds else None
        if not args.only_class and not changes:
            sys.exit("--from-zip меняет только конфиг: задайте --only-class или --thresholds")
        out = Path(args.out)
        digest = variant_from_zip(Path(args.from_zip), out, args.only_class or None, changes)
        what = [f"перепись {args.only_class}"] if args.only_class else []
        what += [f"порогов {sum(len(c) for c in changes.values())}"] if changes else []
        print(f"архив: {out} — {', '.join(what)} из {args.from_zip}, "
              f"{out.stat().st_size / 2**20:.1f} МБ, sha256 {digest}")
        return
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
    config_text = None
    if args.only_class:
        config_text = toml_with_only_classes(
            (ROOT / "configs" / "default.toml").read_text(encoding="utf-8"), args.only_class)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            name = f.relative_to(ROOT).as_posix()
            if config_text is not None and name == "configs/default.toml":
                z.writestr(name, config_text)
            else:
                z.write(f, name)
    size = out.stat().st_size
    if size > LIMIT:
        sys.exit(f"архив {size / 2**20:.0f} МБ больше 2 ГБ")
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    print(f"архив: {out} — {len(files)} файлов, {size / 2**20:.1f} МБ, sha256 {digest}")
    for f in files:
        print("  ", f.relative_to(ROOT).as_posix())


if __name__ == "__main__":
    main()
