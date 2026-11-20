#!/usr/bin/env python3
"""Сборка обучающего ядра ConvNeXt-лидара (варианты c5-in и c12-in) из шаблона `convnext_main.py`.

Курганы лидара финальной версии дают десять сетей: пять фолдов семей c5-in и пять c12-in
(`configs/default.toml`, `[li_cnx]`). Их обучали ядра Kaggle, собранные 05.11 из версии обучающего ядра 01.11
правками 0–9. Этот сборщик повторяет те же
правки на шаблоне `training/lidar_convnext/convnext_main.py`:

0. Встроенные константы на месте `# @@EMBED@@`: режим, голова, задания, участки, фолды семей
   (`folds_li_59.json` — 59 участков с лидаром, пять фолдов) и вендор `VENDOR_B64` — tar.gz файлов из
   `vendor/vendor_manifest.json`. Каждый файл сверяется с манифестом по sha256; код вендора тот же, на котором
   обучались сети.
1. Защита от гонки fork + filelock: spawn, `CNX_WORKERS=0`, кэши HF и timm в /tmp — как в ядрах 05.11.
2. Число итераций — `--iters` (12 000).
3–5. Разметка Li из файла `gt_train_v2.geojson` (EPSG:3857, объекты `markup_type = li`) без S3. Если файла нет,
   ядро, как и ядра 05.11, читает LAS и разметку из S3 организаторов (ключи — в переменных окружения
   `KOZ3_S3_ACCESS_KEY_ID`, `KOZ3_S3_SECRET_ACCESS_KEY`).
6–9. Только c12: задания `c12-*`, двенадцать тегов каналов, масштаб высоты и для каналов 5–9, поиск готовых
   каналов по тегам манифеста.

Каждая правка применяется ровно один раз, иначе сборка падает. Каналы c12 считает `prep12.py`; первые пять
каналов побайтно равны входу c5 (тот же вендор).

    python training/lidar_convnext/make_kernel.py --jobs c5-in:0 --owner OWNER --out BUILD
    python training/lidar_convnext/make_kernel.py --jobs c12-in:0 --owner OWNER --prep-dataset OWNER/koz3-cnx-li-prep12 --out BUILD

Собранное ядро запускается на Kaggle (`kaggle kernels push -p BUILD/<slug>`) или на любой машине с GPU:

    CNX_GT_FILE=work/gt_train_v2.geojson CNX_PREP_DIR=work/cnx_relief12 CNX_OUT=work/c12_f0 python BUILD/<slug>/<slug>.py
"""
from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import io
import json
import re
import sys
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TEMPLATE = HERE / "convnext_main.py"
FOLDS_FILE = HERE / "folds_li_59.json"
VENDOR_MANIFEST = HERE / "vendor" / "vendor_manifest.json"
EMBED = "# @@EMBED@@\n"

#: Имена приватных ресурсов Kaggle, на которых 05.11 обучались сети финальной версии; владелец — `--owner`.
#: Разметку для них даёт `python -m expds.gt`, каналы — режим prep ядра (c5) или prep12.py (c12).
GT_DATASET = "koz3-gate-box-20261104"
PREP_KERNEL = "koz3-cnx-li-prep"


def vendor_sources() -> dict[str, Path]:
    """Путь каждого файла вендора (как в манифесте) → файл репозитория."""
    out: dict[str, Path] = {}
    for name in json.loads(VENDOR_MANIFEST.read_text(encoding="utf-8"))["sha256"]:
        if name.startswith("expds/") or name in ("relief_ext.py", "expds_core.abi3.so"):
            out[name] = ROOT / "inference" / "cnx" / "vendor" / name
        elif name == "compute_metrics.py":
            out[name] = ROOT / "materials" / "compute_metrics.py"
        elif name.startswith("centerhead/") or name == "cnx_head.py":
            out[name] = HERE / "vendor" / name
        else:
            raise SystemExit(f"вендор: неизвестный файл манифеста {name}")
    return out


def vendor_b64() -> str:
    """tar.gz вендора обучения в base64. Файлы сверяются с манифестом; архив детерминирован."""
    manifest = json.loads(VENDOR_MANIFEST.read_text(encoding="utf-8"))
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.PAX_FORMAT) as tf:
        members = sorted(vendor_sources().items()) + [("vendor_manifest.json", VENDOR_MANIFEST)]
        for name, path in members:
            data = path.read_bytes()
            if name != "vendor_manifest.json" and hashlib.sha256(data).hexdigest() != manifest["sha256"][name]:
                raise SystemExit(f"вендор: {path} не совпал с vendor_manifest.json ({name}) — это не код обучения")
            info = tarfile.TarInfo(name)
            info.size, info.mode, info.mtime = len(data), 0o644, 0
            tf.addfile(info, io.BytesIO(data))
    return base64.b64encode(gzip.compress(buf.getvalue(), mtime=0)).decode("ascii")


# ── Правка 1: защита от гонки fork + filelock, как в ядрах 05.11 ─────────────────────────────────────────────
LOG_ANCHOR = 'def log(*a): print(f"[{time.time() - T0:7.0f}s]", *a, flush=True)\n'
ANTIFORK = '''
# ── Патч ядра (make_kernel.py, правка 1): анти-гонка os.fork + filelock (Python 3.13 / filelock) ─────────────
# DataLoader fork при удержании filelock кэша HF/timm другой полосой ронял ядро: spawn вместо fork,
# кэши в /tmp, workers=0 (ожидание данных на T4 около нуля).
import multiprocessing as _mp
try:
    _mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass
os.environ.setdefault("CNX_WORKERS", "0")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HOME", "/tmp/hf_home")
os.environ.setdefault("TORCH_HOME", "/tmp/torch_home")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg_cache")
os.environ.setdefault("TIMM_HOME", "/tmp/timm_home")
try:
    import torch.multiprocessing as _tmp
    _tmp.set_sharing_strategy("file_system")
except Exception:
    pass
'''

# ── Правки 3–5: разметка Li из файла без S3 (как в ядрах 05.11) ────────────────────────────────
HELPERS = '''# ── Разметка Li без S3 (make_kernel.py, правка 3) ─────────────────────────────────────────────────────────────
import unicodedata as _ud
GT_CLS = {c: i for i, c in enumerate(CLASSES)}
def _nfc(s): return _ud.normalize("NFC", str(s or ""))   # имена участков с «ё»/«й»: NFC с обеих сторон
def _find_gt_file():
    p = os.environ.get("CNX_GT_FILE")
    if p and Path(p).is_file(): return Path(p)
    for q in sorted(glob.glob("/kaggle/input/**/gt_train_v2.geojson", recursive=True)): return Path(q)
    return None
def _load_gt_file(path):
    """{участок: [(геометрия EPSG:3857, индекс класса)]} по объектам markup_type = li; плюс счётчик пропусков."""
    from shapely.geometry import shape as _shape
    d = json.load(open(path, encoding="utf-8"))
    n = ((d.get("crs") or {}).get("properties") or {}).get("name", "")
    if n and "3857" not in n: raise SystemExit(f"{path}: ожидается EPSG:3857, в файле {n}")
    out, skipped = collections.defaultdict(list), collections.Counter()
    for f in d.get("features", []):
        pr = f.get("properties") or {}; g = f.get("geometry")
        if pr.get("markup_type") != "li" or not g: continue
        c = GT_CLS.get(pr.get("class_name"))
        if c is None: skipped[str(pr.get("class_name"))] += 1; continue
        try: out[_nfc(pr.get("region_name"))].append((_shape(g), c))
        except Exception: skipped["bad_geometry"] += 1
    return dict(out), dict(skipped)
'''

S3_OLD = '''CLI = s3_client()
try: CLI.list_objects_v2(Bucket=BUCKET, MaxKeys=1)
except Exception as e: raise SystemExit(f"S3 недоступен ({type(e).__name__}): проверьте секреты/интернет (детали скрыты).")
log("S3 OK")
'''
S3_NEW = '''GT_FILE = _find_gt_file()
GT_FEATS = {}
if GT_FILE is None:
    CLI = s3_client()
    try: CLI.list_objects_v2(Bucket=BUCKET, MaxKeys=1)
    except Exception as e: raise SystemExit(f"S3 недоступен ({type(e).__name__}): проверьте секреты/интернет (детали скрыты).")
    log("S3 OK")
else:
    if MODE == "prep": raise SystemExit("prep без S3 невозможен: нужен LAS. Каналы — из выхода prep-ядра")
    CLI = None
    GT_FEATS, _gt_skip = _load_gt_file(GT_FILE)
    log("S3 не нужен: разметка Li из", GT_FILE, "—", sum(len(v) for v in GT_FEATS.values()), "объектов на",
        len(GT_FEATS), "участках; пропущено:", _gt_skip)
'''

INV_OLD = '''with cf.ThreadPoolExecutor(16) as ex: INV = {i["region"]: i for i in ex.map(region_inv, POOL)}
_nolas = [r for r, i in INV.items() if not i["las"]]
if _nolas: raise SystemExit(f"нет LAS в S3: {_nolas}")
log(f"S3-инвентарь: {len(INV)} участков, LAS {sum(i['las'][1] for i in INV.values()) / 1e9:.2f} ГБ, "
    f"файлов разметки Li {sum(len(i['marks']) for i in INV.values())}")
'''
INV_NEW = '''if CLI is not None:
    with cf.ThreadPoolExecutor(16) as ex: INV = {i["region"]: i for i in ex.map(region_inv, POOL)}
    _nolas = [r for r, i in INV.items() if not i["las"]]
    if _nolas: raise SystemExit(f"нет LAS в S3: {_nolas}")
    log(f"S3-инвентарь: {len(INV)} участков, LAS {sum(i['las'][1] for i in INV.values()) / 1e9:.2f} ГБ, "
        f"файлов разметки Li {sum(len(i['marks']) for i in INV.values())}")
else:
    INV = {r: dict(region=r, las=None, marks=[], utm=None) for r in POOL}
    _nogt = [r for r in POOL if not GT_FEATS.get(_nfc(r))]
    log(f"инвентарь без S3: {len(INV)} участков; без разметки Li в файле: {len(_nogt)} {_nogt[:10]}")
    if _nogt and os.environ.get("CNX_ALLOW_NOGT", "0") != "1":   # стоп: вероятно, имена участков не совпали
        raise SystemExit(f"нет разметки Li в {GT_FILE} для {len(_nogt)} участков: {_nogt[:10]}")
'''

LOADGT_OLD = '''def load_gt(region):
    m = META[region]; out = []
'''
LOADGT_NEW = '''def load_gt(region):
    m = META[region]; out = []
    if CLI is None:                       # make_kernel.py, правка 5: разметка из файла (EPSG:3857) → CRS каналов
        tr = Transformer.from_crs("EPSG:3857", m["crs"], always_xy=True).transform
        for g, c in GT_FEATS.get(_nfc(region), []):
            try: g = stf(tr, g)
            except Exception: continue
            if not g.is_valid: g = g.buffer(0)
            if not g.is_empty: out.append((g, c))
        return out
'''

ITERS_OLD = 'iters=E("CNX_ITERS", 10 if SMOKE else 12000)'

# ── Правки 6–9: c12 — двенадцать каналов рельефа поверх c5 (как в ядрах 05.11) ─────────────────
C12_TAGS = ("lrm_s", "lrm_l", "slope", "svf_r", "opp_r", "opn_r", "tpi5_r", "tri3_r", "curv_r", "slrm8_r", "dens_r", "int_r")
PARSE_OLD = ('        assert ch in ("c3", "c5") and ini in ("in", "sc"), tok\n'
             '        out.append(dict(variant=v, ch=int(ch[1]), init=ini, fold=int(f)))\n')
PARSE_NEW = ('        assert ch in ("c3", "c5", "c12") and ini in ("in", "sc"), tok\n'
             '        out.append(dict(variant=v, ch=int(ch[1:]), init=ini, fold=int(f)))\n')
TAGS_OLD = 'TAGS5 = ("lrm_s", "lrm_l", "slope", "svf_r", "opp_r")   # порядок каналов = контракт модели (c3 = первые три)\n'
TAGS_NEW = f"TAGS5 = {C12_TAGS!r}   # c12 (make_kernel.py --prep-dataset): порядок каналов = контракт, первые пять = c5\n"
# масштаб высоты ×h — для всех каналов рельефа (5–9: открытость −, TPI, TRI, кривизна, SLRM); плотность и интенсивность — нет
HEIGHT_OLD = '            for c in [c for c in (0, 1, 3, 4) if c < self.nch]: img[c] = 128 + (img[c] - 128) * h\n'
HEIGHT_NEW = '            for c in [c for c in (0, 1, 3, 4, 5, 6, 7, 8, 9) if c < self.nch]: img[c] = 128 + (img[c] - 128) * h\n'
# Kaggle распаковывает zip датасета без папки cnx_relief: c12 ищет manifest.json с тегами TAGS5 в любом месте /kaggle/input
FINDPREP_OLD = ('    for p in sorted(glob.glob("/kaggle/input/**/cnx_relief/manifest.json", recursive=True)):\n'
                '        return Path(p).parent\n')
FINDPREP_NEW = ('    for p in sorted(glob.glob("/kaggle/input/**/manifest.json", recursive=True)):   # c12: датасет без папки cnx_relief\n'
                '        try: _m = json.load(open(p, encoding="utf-8"))\n'
                '        except Exception: continue\n'
                '        if isinstance(_m, dict) and _m.get("tags") == list(TAGS5) and _m.get("regions"): return Path(p).parent\n')

JOB_RE = re.compile(r"^(c3|c5|c12)-(in|sc):[0-4]$")


def _replace_once(text: str, old: str, new: str, what: str) -> str:
    n = text.count(old)
    if n != 1:
        raise SystemExit(f"правка «{what}»: якорь найден {n} раз(а), нужен ровно один — шаблон изменился?")
    return text.replace(old, new)


def embed_block(jobs: str, mode: str, regions: str) -> str:
    folds = FOLDS_FILE.read_text(encoding="utf-8").strip()
    json.loads(folds)
    if "'''" in folds:
        raise SystemExit("folds_li_59.json не встраивается в r'''…'''")
    return (f"CNX_MODE_DEFAULT = {mode!r}\n"
            "CNX_HEAD_DEFAULT = 'centerhead'\n"
            f"CNX_JOBS_DEFAULT = {jobs!r}\n"
            f"CNX_REGIONS_DEFAULT = {regions!r}\n"
            f"FOLDS = __import__('json').loads(r'''{folds}''')\n"
            f"VENDOR_B64 = '{vendor_b64()}'\n")


def build(jobs: str, iters: int = 12000, mode: str = "full", regions: str = "", prep_dataset: str = "",
          template: Path = TEMPLATE) -> str:
    """Текст ядра. mode smoke (10 итераций) и regions (подмножество участков) — только для проверки пути;
    c12 — только с prep_dataset (каналы prep12.py) и без смешения с c3/c5 в одном ядре."""
    if mode not in ("full", "smoke", "prep"):
        raise SystemExit(f"mode {mode!r}: full, smoke или prep")
    if regions and mode != "smoke":
        raise SystemExit("--regions только для smoke: веса и пороги требуют OOF по всем участкам")
    toks = [t.strip() for t in jobs.split(",") if t.strip()]
    if not toks:
        raise SystemExit("--jobs пуст")
    for tok in toks:
        if not JOB_RE.match(tok):
            raise SystemExit(f"задание {tok!r}: ожидается c3|c5|c12-in|sc:0..4")
    c12 = [t.startswith("c12-") for t in toks]
    if any(c12) and not all(c12):
        raise SystemExit("c12 и c3/c5 в одном ядре не смешиваются: у них разные входы")
    if any(c12) and not prep_dataset:
        raise SystemExit("c12: нужен --prep-dataset (каналы prep12.py: cnx_relief с 12 каналами)")
    if prep_dataset and not any(c12):
        raise SystemExit("--prep-dataset только для c12: c3/c5 читают выход prep-ядра")
    if any(c12) and mode == "prep":
        raise SystemExit("каналы c12 строит prep12.py, а не режим prep ядра")
    text = template.read_text(encoding="utf-8")
    text = _replace_once(text, EMBED, embed_block(",".join(toks), mode, regions), "0 встроенные константы")
    text = _replace_once(text, LOG_ANCHOR, LOG_ANCHOR + ANTIFORK, "1 анти-гонка fork")
    text = _replace_once(text, ITERS_OLD, f'iters=E("CNX_ITERS", 10 if SMOKE else {int(iters)})', "2 итерации")
    text = _replace_once(text, S3_OLD, HELPERS + S3_NEW, "3 разметка без S3")
    text = _replace_once(text, INV_OLD, INV_NEW, "4 инвентарь без S3")
    text = _replace_once(text, LOADGT_OLD, LOADGT_NEW, "5 load_gt из файла")
    if any(c12):
        text = _replace_once(text, PARSE_OLD, PARSE_NEW, "6 задания c12")
        text = _replace_once(text, TAGS_OLD, TAGS_NEW, "7 теги c12")
        text = _replace_once(text, HEIGHT_OLD, HEIGHT_NEW, "8 высота c12")
        text = _replace_once(text, FINDPREP_OLD, FINDPREP_NEW, "9 поиск каналов c12")
    header = (f"# ── Сборка training/lidar_convnext/make_kernel.py: задания {','.join(toks)}; режим {mode}; "
              f"итераций {iters if mode == 'full' else 10}; каналы {prep_dataset or PREP_KERNEL} ──\n")
    return header + text


def kernel_metadata(slug: str, owner: str, c12: bool, prep_dataset: str, gt_dataset: str) -> dict:
    """kernel-metadata.json для `kaggle kernels push`: приватное ядро с GPU и интернетом (pip-пины)."""
    meta = dict(id=f"{owner}/{slug}", title=slug, code_file=f"{slug}.py", language="python", kernel_type="script",
                is_private=True, enable_gpu=True, enable_internet=True, dataset_sources=[gt_dataset],
                kernel_sources=[f"{owner}/{PREP_KERNEL}"], competition_sources=[])
    if c12:
        meta.update(dataset_sources=[gt_dataset, prep_dataset], kernel_sources=[])
    return meta


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--jobs", required=True, help="например c5-in:0 или c12-in:3 (несколько — через запятую)")
    ap.add_argument("--out", type=Path, required=True, help="каталог сборки (вне git)")
    ap.add_argument("--slug", default="", help="имя ядра на Kaggle; по умолчанию koz3-li-cnx-<задания>")
    ap.add_argument("--iters", type=int, default=12000)
    ap.add_argument("--mode", default="full", choices=("full", "smoke", "prep"))
    ap.add_argument("--regions", default="", help="только smoke: участки через запятую")
    ap.add_argument("--prep-dataset", default="", help="только c12: датасет Kaggle с выходом prep12.py")
    ap.add_argument("--owner", required=True, help="учётная запись Kaggle, от имени которой пушится ядро")
    ap.add_argument("--gt-dataset", default="", help="датасет Kaggle с gt_train_v2.geojson; по умолчанию OWNER/"
                    + GT_DATASET)
    a = ap.parse_args(argv)
    slug = a.slug or "koz3-li-cnx-" + re.sub(r"[^a-z0-9]+", "-", a.jobs.lower()).strip("-")
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{4,49}", slug):
        ap.error("slug: строчная латиница, цифры, дефисы, 5–50 символов")
    text = build(a.jobs, a.iters, a.mode, a.regions, a.prep_dataset)
    d = a.out / slug
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{slug}.py").write_text(text, encoding="utf-8")
    meta = kernel_metadata(slug, a.owner, a.jobs.strip().startswith("c12-"), a.prep_dataset,
                           a.gt_dataset or f"{a.owner}/{GT_DATASET}")
    (d / "kernel-metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"собрано: {d} (задания {a.jobs}, режим {a.mode}, sha256 ядра "
          f"{hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
