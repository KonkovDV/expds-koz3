#!/usr/bin/env python3
"""Быстрая воспроизводимая сборка архива КОЗ №3 под эксперимент (bot-8).

Файлы берутся из git-коммита (--ref, по умолчанию HEAD), а не из рабочей папки:
незакоммиченные правки и чужие черновики в архив не попадают, один и тот же ввод
даёт байт-в-байт один и тот же zip (фиксированные даты, порядок, права).

    python scripts/build_submission_git.py --repo . --label base                     # HEAD как есть
    python scripts/build_submission_git.py --repo . --label thr2 --config my.toml    # свой default.toml
    python scripts/build_submission_git.py --repo . --label newli \
        --weight models/li_r18_hard2.pt=/path/li_new.pt                     # подмена весов
    python scripts/build_submission_git.py --repo . --label kur --only-classes kurgany_tselye kurgany_povrezhdennye
    python build_submission.py ... --check                                  # сразу check_submission.py

Имя: <out-dir>/koz3_<label>_<git7>_<cfg8>[_w<wts8>].zip + .manifest.json (sha256 каждого файла).
В архив: metadata.json, inference/main.py, inference/lib/expds_core*.so, expds/**/*.py,
configs/default.toml (единственный конфиг, который читает main.py:86) и только веса,
на которые ссылается этот конфиг. .pt/.so кладутся без сжатия (ZIP_STORED) — быстрее, размер почти тот же.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
import tomllib
import zipfile
from pathlib import Path, PurePosixPath

CLASSES = ("kurgany_povrezhdennye", "kurgany_tselye", "gorodishcha", "arkhitektury", "fortifikatsii",
           "object_poly", "finds_points")  # expds/constants.py

LIMIT = 2 * 1024 ** 3
FORBIDDEN = {".las", ".laz", ".tif", ".tiff", ".geojson", ".npy", ".npz", ".csv", ".jpg", ".png", ".jp2",
             ".shp", ".gpkg", ".zip", ".ipynb", ".pkl"}
STORED = {".pt", ".pth", ".so", ".onnx", ".ckpt", ".bin", ".whl"}
EPOCH = (1980, 1, 1, 0, 0, 0)


def git(repo: Path, *args: str, binary: bool = False):
    r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, check=False)
    if r.returncode:
        sys.exit(f"git {' '.join(args)}: {r.stderr.decode(errors='replace').strip()}")
    return r.stdout if binary else r.stdout.decode().strip()


def w5cf_refs(cfg: dict) -> list[str]:
    """W4 v4: артефакты RR1s-cf (W1 expds.w5cf_infer) при w5_cloudfeat=true: ref.npz шкалы бленда + каталог ранжировщиков.
    Файлы fam<k>.txt добавляются из meta.json в main() (список моделей знает только meta)."""
    if not cfg.get("w5_cloudfeat"):
        return []
    s = cfg.get("w5cf") or {}
    d = str(s.get("infer_dir", "models/w5cf_infer")).rstrip("/")
    return [s.get("ref", "models/w5cf/ref.npz"), f"{d}/meta.json", f"{d}/gq_ref.npz"]


def w5cf_npz_ok(path: str, data: bytes) -> bool:
    """.npz запрещены (данные), кроме двух таблиц RR1s-cf: только 1-D числовые массивы/ключи, без координат/меток."""
    import io
    import numpy as np
    z = np.load(io.BytesIO(data), allow_pickle=False)
    bad = {"x", "y", "lon", "lat", "geometry", "label", "hit", "gt", "region", "reg"}
    return all(z[k].ndim == 1 and z[k].dtype.kind in "biuf" and k.split("__")[0].lower() not in bad for k in z.files)


def model_refs(cfg: dict) -> list[str]:
    """Те же правила, что scripts/build_submission.py:38-53 (model_specs/model_files), без импорта expds."""
    out = []
    for spec in (cfg.get("models") or {}).values():
        for it in spec if isinstance(spec, list) else ([spec] if spec else []):
            path = it if isinstance(it, str) else (it.get("path") if isinstance(it, dict) else None)
            out += path if isinstance(path, list) else ([path] if path else [])
    lg = (cfg.get("candidates") or {}).get("lgbm_model")
    v = cfg.get("verify") or {}
    out += [lg] if lg else []
    out += [v["weights"]] if v.get("enabled") and v.get("weights") else []
    rr = cfg.get("rerank_arch") or {}   # bot-7 (bot7-candidate): модель реранка архитектуры, если флаг включён
    out += [rr.get("model", "models/rerank_arch_lgbm.txt"), rr.get("meta", "models/rerank_arch_lgbm.json")] if rr.get("enabled") else []
    # bot-8 bot8-cnx: веса [li_cnx] сюда не входят — их берёт cnx_weights() по WEIGHTS_MANIFEST (не из git)
    pix = (cfg.get("geophys") or {}).get("pixgbm") or {}
    if pix.get("enabled") and pix.get("impl") == "clean":   # W4: модели clean-room pix (legacy PIXGBM не собирается)
        md = str(pix.get("model_dir", "models/pix_clean")).rstrip("/")
        out += [f"{md}/poly.txt", f"{md}/finds.txt", f"{md}/meta.json"]
        if pix.get("org_etgr"):   # W4 GEO-ORG3
            out += [f"{md}/org/org_etgr_map.txt", f"{md}/org/org_etgr_prof.txt", f"{md}/org/org_etgr_meta.json"]
    out += w5cf_refs(cfg)
    sd = cfg.get("sp_dfine") or {}
    if sd.get("enabled"):   # W4 v3: D-FINE SpOr — каталоги HF (config.json + model.safetensors), как build_submission.py
        out += [f"{str(w).rstrip('/')}/{f}" for w in sd.get("weights", []) for f in ("config.json", "model.safetensors")]
    seen, uniq = set(), []
    for p in out:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq


def cnx_enabled(cfg: dict) -> bool:
    lc = cfg.get("li_cnx") or {}
    return lc.get("enabled") is True and bool(lc.get("weights"))   # только bool, как expds/lidar_cnx.enabled


CNX_WEIGHTS_MANIFEST = "inference/cnx/WEIGHTS_MANIFEST.json"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cnx_weights(cfg: dict, wman: dict, overrides: dict, tracked: set) -> dict[str, Path]:
    """bot-8: веса [li_cnx] → {путь в архиве: локальный файл}. Только из WEIGHTS_MANIFEST.json: файл —
    --weight или поле source; sha256 и size обязаны совпасть, sha из never_pack (smoke-веса) — отказ.
    Веса cnx в git не держим: путь, отслеживаемый git, — отказ."""
    w = (cfg.get("li_cnx") or {}).get("weights")
    want = [w] if isinstance(w, str) else list(w or [])
    entries = {e.get("path"): e for e in wman.get("weights") or []}
    never = {e.get("sha256") for e in wman.get("never_pack") or []}
    out = {}
    for rel in want:
        e = entries.get(rel)
        if e is None:
            sys.exit(f"[li_cnx] {rel}: нет в {CNX_WEIGHTS_MANIFEST} — веса cnx пакуются только по манифесту")
        if rel in tracked:
            sys.exit(f"[li_cnx] {rel} отслеживается git — веса cnx в репозиторий не кладём")
        src = overrides.get(rel) or Path(str(e.get("source") or ""))
        if not src.is_file():
            sys.exit(f"[li_cnx] {rel}: нет файла {src} (source манифеста или --weight)")
        size, sha = src.stat().st_size, _sha256_file(src)
        if sha in never:
            sys.exit(f"[li_cnx] {rel}: {src} — smoke-веса из never_pack, паковать нельзя")
        if size != int(e.get("size", -1)) or sha != e.get("sha256"):
            sys.exit(f"[li_cnx] {rel}: {src} не совпадает с манифестом (size {size} vs {e.get('size')}, "
                     f"sha256 {sha[:12]} vs {str(e.get('sha256'))[:12]})")
        out[rel] = src
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Сборка архива КОЗ №3 из git-коммита (воспроизводимо)")
    ap.add_argument("--repo", type=Path, default=Path("repo"), help="git-репозиторий expds-koz3")
    ap.add_argument("--ref", default="HEAD", help="коммит/ветка (HEAD)")
    ap.add_argument("--label", required=True, help="метка эксперимента в имени файла [a-z0-9_-]")
    ap.add_argument("--config", type=Path, help="свой TOML вместо configs/default.toml из коммита")
    ap.add_argument("--only-classes", nargs="+", help="дописать [post] only_classes (перепись классов)")
    ap.add_argument("--weight", action="append", default=[], metavar="ARCHIVE_PATH=LOCAL_FILE",
                    help="подменить/добавить файл весов (можно несколько раз)")
    ap.add_argument("--so", type=Path, help="свой expds_core.abi3.so вместо закоммиченного")
    ap.add_argument("--vendor", type=Path, help="каталог распакованных колёс (transformers 5.21.0 …) → inference/lib/vendor/; "
                    "обязателен при [sp_dfine]/[gor_gate] enabled")
    ap.add_argument("--out-dir", type=Path, default=Path("dist"))
    ap.add_argument("--check", action="store_true", help="после сборки запустить check_submission.py")
    ap.add_argument("--check-args", default="", help="доп. аргументы check_submission.py, одной строкой")
    args = ap.parse_args()
    t0 = time.monotonic()
    if not all(c.isalnum() or c in "_-" for c in args.label):
        sys.exit("label: только буквы, цифры, _ и -")
    repo = args.repo.resolve()
    sha = git(repo, "rev-parse", args.ref)
    tracked = git(repo, "ls-tree", "-r", "--name-only", sha).splitlines()

    def blob(path: str) -> bytes:
        return git(repo, "show", f"{sha}:{path}", binary=True)

    files: dict[str, bytes] = {}
    pick = ["metadata.json", "inference/main.py"]
    pick += [p for p in tracked if p.startswith("inference/lib/expds_core") and p.endswith(".so")]
    # W4 DOC-1: файлы, производные от EN, в архив не идут (scripts/check_archive_clean.py)
    pick += [p for p in tracked if p.startswith("expds/") and p.endswith(".py")
             and PurePosixPath(p).name not in {"geo_pixgbm.py", "geo_finds_gbm.py", "pix_legacy.py", "geo_backbone.py"}]
    for p in pick:
        files[p] = blob(p)
    if args.so:
        for p in [p for p in files if p.startswith("inference/lib/")]:
            del files[p]
        files[f"inference/lib/{args.so.name}"] = args.so.read_bytes()
    cfg_text = args.config.read_text(encoding="utf-8") if args.config else blob("configs/default.toml").decode("utf-8")
    if args.only_classes:
        bad = sorted(set(args.only_classes) - set(CLASSES))
        if bad:
            sys.exit(f"нет таких классов: {bad}; допустимы {CLASSES}")
        line = "only_classes = [" + ", ".join(json.dumps(c) for c in args.only_classes) + "]"
        post = tomllib.loads(cfg_text).get("post")  # строка «# [post]» в комментарии — не секция
        if post is None:
            cfg_text = cfg_text.rstrip() + "\n\n[post]\n" + line + "\n"
        elif "only_classes" in post:
            sys.exit("в [post] уже есть only_classes — правьте конфиг вручную (--config)")
        else:  # ключ сразу под заголовком секции, чтобы не попасть в подтаблицу
            cfg_text, n = re.subn(r"(?m)^\[post\][ \t]*$", "[post]\n" + line, cfg_text, count=1)
            if n != 1:
                sys.exit("не нашёл заголовок [post] — правьте конфиг вручную (--config)")
    try:
        cfg = tomllib.loads(cfg_text)
    except tomllib.TOMLDecodeError as exc:
        sys.exit(f"конфиг не TOML: {exc}")
    files["configs/default.toml"] = cfg_text.encode("utf-8")
    if cnx_enabled(cfg):   # bot-8 bot8-cnx: вендор подготовки bot-9 (cnx_prep.py, vendor/, _deps/rvt, колесо rvt-py)
        cnx = [p for p in tracked if p.startswith("inference/cnx/") and "/__pycache__/" not in p]
        if "inference/cnx/cnx_prep.py" not in cnx:
            sys.exit("[li_cnx] включён, но в коммите нет inference/cnx/ (вендор подготовки cnx)")
        for p in cnx:
            files[p] = blob(p)
        if CNX_WEIGHTS_MANIFEST not in files:
            sys.exit(f"[li_cnx] включён, но в коммите нет {CNX_WEIGHTS_MANIFEST}")
        if str((cfg.get("li_cnx") or {}).get("decode") or "rescore") == "head":   # 05.11: expds/lidar_cnx_head.py
            ch = [p for p in tracked if p.startswith("centerhead/") and p.count("/") == 1 and p.endswith(".py")]
            if "centerhead/decode.py" not in ch:
                sys.exit("[li_cnx] decode = \"head\", а в коммите нет пакета centerhead/")
            for p in ch:
                files[p] = blob(p)
    need_vendor = any((cfg.get(sec) or {}).get("enabled") for sec in ("sp_dfine", "gor_gate"))
    if need_vendor and not (args.vendor and args.vendor.is_dir()):
        sys.exit("[sp_dfine]/[gor_gate] enabled — нужен --vendor <каталог распакованных колёс>")
    if args.vendor and need_vendor:
        vroot = args.vendor.resolve()
        for q in sorted(vroot.rglob("*")):
            if q.is_file() and "__pycache__" not in q.parts and q.suffix != ".pyc":
                files["inference/lib/vendor/" + q.relative_to(vroot).as_posix()] = q.read_bytes()
    meta = json.loads(files["metadata.json"])
    if not meta.get("image") or "<" in str(meta.get("image")) or not meta.get("entry_point"):
        sys.exit(f"metadata.json не заполнен: {meta}")
    overrides = {}
    for spec in args.weight:
        dst, _, src = spec.partition("=")
        if not src or not Path(src).is_file():
            sys.exit(f"--weight {spec}: нужен ARCHIVE_PATH=существующий_файл")
        overrides[dst] = Path(src)
    missing = []
    for rel in model_refs(cfg):
        if rel in overrides:
            files[rel] = overrides[rel].read_bytes()
        elif rel in tracked:
            files[rel] = blob(rel)
        else:
            missing.append(rel)
    w5npz = set()
    if cfg.get("w5_cloudfeat") and not missing:   # W4 v4: модели ранжировщика из meta.json
        refs = w5cf_refs(cfg); mrel = refs[1]; d = mrel.rsplit("/", 1)[0]
        w5npz = {r for r in refs if r.endswith(".npz")}
        for v in json.loads(files[mrel])["models"].values():
            rel = f"{d}/{v['file']}"
            if rel in overrides:
                files[rel] = overrides[rel].read_bytes()
            elif rel in tracked:
                files[rel] = blob(rel)
            else:
                missing.append(rel)
        for r in w5npz:
            if not w5cf_npz_ok(r, files[r]):
                sys.exit(f"w5cf: {r} — не таблица 1-D (запрещено как данные)")
        # sha256 артефактов ранжировщика в архиве: проверяются w5_cloudfeat.preflight до начала работы
        w5 = sorted({*refs, *(f"{d}/{v['file']}" for v in json.loads(files[mrel])["models"].values())})
        files["WEIGHTS_MANIFEST.json"] = json.dumps(
            {"w5cf": {r: {"sha256": hashlib.sha256(files[r]).hexdigest(), "bytes": len(files[r])} for r in w5}},
            ensure_ascii=False, indent=1, sort_keys=True).encode("utf-8")
    if missing:
        sys.exit(f"веса из конфига не найдены ни в коммите, ни в --weight: {missing}")
    cnx_src = cnx_weights(cfg, json.loads(files[CNX_WEIGHTS_MANIFEST]), overrides, set(tracked)) \
        if cnx_enabled(cfg) else {}
    for rel, src in cnx_src.items():
        files[rel] = src.read_bytes()
        print(f"[li_cnx] {rel} ← {src} (sha256 и size = WEIGHTS_MANIFEST)")
    unused = sorted(set(overrides) - set(files))
    if unused:
        print(f"ВНИМАНИЕ: --weight {unused} не упомянуты в конфиге — в архив не попадут", file=sys.stderr)
    bad = [p for p in files if PurePosixPath(p).suffix.lower() in FORBIDDEN and p not in w5npz]
    if bad:
        sys.exit(f"запрещённые файлы: {bad}")
    dirty = git(repo, "status", "--porcelain", "--untracked-files=no", "--", *sorted(set(pick) & set(tracked)))
    cfg_sha = hashlib.sha256(files["configs/default.toml"]).hexdigest()
    w_sha = hashlib.sha256(b"".join(k.encode() + hashlib.sha256(v.read_bytes()).digest()
                                    for k, v in sorted(overrides.items()))).hexdigest() if overrides else ""
    name = f"koz3_{args.label}_{sha[:7]}_{cfg_sha[:8]}" + (f"_w{w_sha[:8]}" if w_sha else "") + ".zip"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / name
    tmp = out.with_suffix(".zip.part")
    with zipfile.ZipFile(tmp, "w") as z:
        for p in sorted(files):
            zi = zipfile.ZipInfo(p, EPOCH)
            zi.external_attr = 0o644 << 16
            zi.create_system = 3
            stored = PurePosixPath(p).suffix in STORED
            zi.compress_type = zipfile.ZIP_STORED if stored else zipfile.ZIP_DEFLATED
            z.writestr(zi, files[p], compresslevel=None if stored else 6)
    size = tmp.stat().st_size
    if size > LIMIT:
        tmp.unlink()
        sys.exit(f"архив {size / 2**20:.0f} МиБ > 2048 МиБ — уберите веса из конфига")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from check_archive_clean import check as doc1_check   # W4 DOC-1
    viol = doc1_check(tmp)
    if viol:
        tmp.unlink()
        sys.exit("DOC-1: архив не чистый:\n" + "\n".join(viol))
    tmp.replace(out)
    zsha = hashlib.sha256(out.read_bytes()).hexdigest()
    manifest = {"zip": name, "sha256": zsha, "bytes": size, "git": sha, "ref": args.ref, "label": args.label,
                "config_sha256": cfg_sha, "config_source": str(args.config) if args.config else f"{sha[:7]}:configs/default.toml",
                "only_classes": args.only_classes, "weight_overrides": {k: str(v) for k, v in overrides.items()},
                "cnx_weights": {k: str(v) for k, v in cnx_src.items()},
                "so_override": str(args.so) if args.so else None,
                "vendor": str(args.vendor) if args.vendor and need_vendor else None,
                "files": {p: hashlib.sha256(b).hexdigest() for p, b in sorted(files.items())},
                "built": time.strftime("%Y-%m-%d %H:%M:%S %Z")}
    out.with_suffix(".manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{out}  {size / 2**20:.1f} МиБ  {len(files)} файлов  sha256 {zsha}  ({time.monotonic() - t0:.1f} с)")
    if dirty:
        print(f"ВНИМАНИЕ: в рабочей папке есть незакоммиченные правки, в архив НЕ вошли:\n{dirty}", file=sys.stderr)
    if args.check:
        chk = Path(__file__).with_name("check_submission.py")
        return subprocess.call([sys.executable, str(chk), str(out), *args.check_args.split()])
    return 0


if __name__ == "__main__":
    sys.exit(main())
