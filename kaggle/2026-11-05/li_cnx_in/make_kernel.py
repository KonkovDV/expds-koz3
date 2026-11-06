#!/usr/bin/env python3
"""Ядро «лидар ConvNeXt-T с инициализацией ImageNet» (c5-in) без S3: сборка из ядра bot-9 (снимок 01.11).

Зачем — `docs/plans/EN_LIDAR_RECIPE_20261104.md` §3–4. Главная непроверенная разница нашей линии cnx и лидара EN
(LB P/R .603/.637 против наших .516/.504) — инициализация ImageNet: вариант `in` не запускался ни разу. Ключей S3
в Kaggle больше нет (датасет `dmitrijkv/koz3-s3-cred` удалён), поэтому разметка и каналы берутся из входов ядра.

Правки к `snapshot/2026-11-01/kaggle_push/koz3-cnx-li-c3sc-f4-v2/koz3-cnx-li-c3sc-f4-v2.py` (sha256 ниже). Каждая
применяется ровно один раз, иначе сборка падает:

1. `CNX_JOBS_DEFAULT` — задания этого ядра (`--jobs`, например `c5-in:0,c5-in:1`; на T4×2 — по одному на GPU).
2. Итераций по умолчанию — `--iters` (PREREG: 12 000, как у c3sc и у EN; при 0,51 с/ит это ~1,7 ч на фолд).
3. Разметка Li — из `gt_train_v2.geojson` (приватный датасет `dmitrijkv/koz3-gate-box-20261104`, объекты с
   `markup_type = li`). Клиент S3 создаётся, только если файла разметки нет — тогда ядро ведёт себя как снимок.
4. Инвентарь участков без S3: LAS не нужен, каналы рельефа — из выхода `dmitrijkv/koz3-cnx-li-prep`
   (`kernel_sources`); вендор каналов не меняется, проверка sha prep-выхода в ядре остаётся.
5. `load_gt`: геометрии файла (EPSG:3857) переводятся в CRS каналов участка.

Режим prep без S3 невозможен (нужен LAS) — ядро останавливается с понятной ошибкой.

    python kaggle/2026-11-05/li_cnx_in/make_kernel.py --slug koz3-li-cnx-in-a --jobs c5-in:0,c5-in:1 --out BUILD
    kaggle kernels push -p BUILD/koz3-li-cnx-in-a    # токен dmitrijkv, только после коммита PREREG.md

Код в git не дублируется (снимок уже в репозитории): собранное ядро кладётся в `--out`, не в дерево.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "snapshot/2026-11-01/kaggle_push/koz3-cnx-li-c3sc-f4-v2/koz3-cnx-li-c3sc-f4-v2.py"
SRC_META = SRC.with_name("kernel-metadata.json")
SRC_SHA256 = "10590f13b35656db459e4020741882a7a99369dd3d8c9dc6336cc0cf9961acca"
GT_DATASET = "dmitrijkv/koz3-gate-box-20261104"
PREP_KERNEL = "dmitrijkv/koz3-cnx-li-prep"
OWNER = "dmitrijkv"

HEADER = """# ── Сборка kaggle/2026-11-05/li_cnx_in/make_kernel.py (PREREG kaggle/2026-11-05/li_cnx_in/PREREG.md) ─────────
# Источник: snapshot/2026-11-01/kaggle_push/koz3-cnx-li-c3sc-f4-v2 (sha256 {sha}). Без S3: разметка Li — gt_train_v2.geojson
# ({gt_ds}), каналы — выход {prep}. Задания: {jobs}; итераций: {iters}. ImageNet (timm convnext_tiny.fb_in1k)
# разрешён организаторами 29.10 и владельцем 31.10 (docs/RULES.md); пометка «только измерение» ниже — из снимка 01.11.
"""

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
    if MODE == "prep": raise SystemExit("prep без S3 невозможен: нужен LAS. Каналы — из выхода dmitrijkv/koz3-cnx-li-prep")
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
    if _nogt and os.environ.get("CNX_ALLOW_NOGT", "0") != "1":   # PREREG: стоп — вероятно, имена не совпали
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

JOB_RE = re.compile(r"^(c3|c5|c12)-(in|sc):[0-4]$")

# ── c12 (05.11, вторая сессия): 12 каналов рельефа — рецепт EN (9–12 каналов) поверх c5. Каналы считает тот же вендор
# bot-9 (relief_ext.EXTRA_TAGS) вне Kaggle из LAS организаторов; первые пять побайтно равны выходу koz3-cnx-li-prep.
# Вход — приватный датасет (--prep-dataset) с cnx_relief/{участок}.npy|.json и manifest.json (теги — C12_TAGS).
C12_TAGS = ("lrm_s", "lrm_l", "slope", "svf_r", "opp_r", "opn_r", "tpi5_r", "tri3_r", "curv_r", "slrm8_r", "dens_r", "int_r")
PARSE_OLD = ('        assert ch in ("c3", "c5") and ini in ("in", "sc"), tok\n'
             '        out.append(dict(variant=v, ch=int(ch[1]), init=ini, fold=int(f)))\n')
PARSE_NEW = ('        assert ch in ("c3", "c5", "c12") and ini in ("in", "sc"), tok\n'
             '        out.append(dict(variant=v, ch=int(ch[1:]), init=ini, fold=int(f)))\n')
TAGS_OLD = 'TAGS5 = ("lrm_s", "lrm_l", "slope", "svf_r", "opp_r")   # порядок каналов = контракт модели (c3 = первые три)\n'
TAGS_NEW = (f"TAGS5 = {C12_TAGS!r}   # c12 (make_kernel.py --prep-dataset): порядок каналов = контракт, первые пять = c5\n")
# масштаб высоты ×h — для всех каналов рельефа (5–9: открытость −, TPI, TRI, кривизна, SLRM); плотность и интенсивность — нет
HEIGHT_OLD = '            for c in [c for c in (0, 1, 3, 4) if c < self.nch]: img[c] = 128 + (img[c] - 128) * h\n'
HEIGHT_NEW = '            for c in [c for c in (0, 1, 3, 4, 5, 6, 7, 8, 9) if c < self.nch]: img[c] = 128 + (img[c] - 128) * h\n'
# zip датасета (kaggle datasets create --dir-mode zip) Kaggle распаковывает без папки cnx_relief (ядра a/b 05.11 упали на 46 с:
# «нет каналов для 001_…»). c12 ищет manifest.json с тегами TAGS5 в любом месте /kaggle/input; c3/c5 — без изменений.
FINDPREP_OLD = ('    for p in sorted(glob.glob("/kaggle/input/**/cnx_relief/manifest.json", recursive=True)):\n'
                '        return Path(p).parent\n')
FINDPREP_NEW = ('    for p in sorted(glob.glob("/kaggle/input/**/manifest.json", recursive=True)):   # c12: датасет без папки cnx_relief\n'
                '        try: _m = json.load(open(p, encoding="utf-8"))\n'
                '        except Exception: continue\n'
                '        if isinstance(_m, dict) and _m.get("tags") == list(TAGS5) and _m.get("regions"): return Path(p).parent\n')


def _replace_once(text: str, old: str, new: str, what: str) -> str:
    n = text.count(old)
    if n != 1:
        raise SystemExit(f"правка «{what}»: якорь найден {n} раз(а), нужен ровно один — снимок ядра изменился?")
    return text.replace(old, new)


def build(slug: str, jobs: str, iters: int, src: Path = SRC, mode: str = "full",
          regions: str = "", prep_dataset: str = "") -> tuple[str, dict]:
    """Текст ядра и kernel-metadata.json. mode smoke (10 итераций) и regions (подмножество участков) — только для
    проверки пути без S3 до полного прогона; в гейт идут лишь ядра mode full по всем участкам.
    c12 — только вместе с prep_dataset (приватный датасет 12 каналов) и без смешения с c3/c5 в одном ядре."""
    if mode not in ("full", "smoke"):
        raise SystemExit(f"mode {mode!r}: full или smoke")
    if regions and mode != "smoke":
        raise SystemExit("--regions только для smoke: гейт требует OOF по всем 59 участкам")
    toks = [t.strip() for t in jobs.split(",")]
    for tok in toks:
        if not JOB_RE.match(tok):
            raise SystemExit(f"задание {tok!r}: ожидается c3|c5|c12-in|sc:0..4")
    c12 = [t.startswith("c12-") for t in toks]
    if any(c12) and not all(c12):
        raise SystemExit("c12 и c3/c5 в одном ядре не смешиваются: у них разные входы")
    if any(c12) and not prep_dataset:
        raise SystemExit("c12: нужен --prep-dataset (приватный датасет cnx_relief с 12 каналами)")
    if prep_dataset and not any(c12):
        raise SystemExit("--prep-dataset только для c12: c3/c5 читают выход koz3-cnx-li-prep")
    raw = src.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if sha != SRC_SHA256:
        raise SystemExit(f"снимок ядра изменился: sha256 {sha} ≠ {SRC_SHA256}")
    text = raw.decode("utf-8")
    text = _replace_once(text, "CNX_JOBS_DEFAULT = 'c3-sc:4'\n", f"CNX_JOBS_DEFAULT = {jobs!r}\n", "1 задания")
    text = _replace_once(text, 'iters=E("CNX_ITERS", 10 if SMOKE else 12000)',
                         f'iters=E("CNX_ITERS", 10 if SMOKE else {int(iters)})', "2 итерации")
    text = _replace_once(text, "CNX_MODE_DEFAULT = 'full'\n", f"CNX_MODE_DEFAULT = {mode!r}\n", "0 режим")
    text = _replace_once(text, "CNX_REGIONS_DEFAULT = ''\n", f"CNX_REGIONS_DEFAULT = {regions!r}\n", "0 участки")
    text = _replace_once(text, S3_OLD, HELPERS + S3_NEW, "3 разметка без S3")
    text = _replace_once(text, INV_OLD, INV_NEW, "4 инвентарь без S3")
    text = _replace_once(text, LOADGT_OLD, LOADGT_NEW, "5 load_gt из файла")
    if any(c12):
        text = _replace_once(text, PARSE_OLD, PARSE_NEW, "6 задания c12")
        text = _replace_once(text, TAGS_OLD, TAGS_NEW, "7 теги c12")
        text = _replace_once(text, HEIGHT_OLD, HEIGHT_NEW, "8 высота c12")
        text = _replace_once(text, FINDPREP_OLD, FINDPREP_NEW, "9 поиск каналов c12")
    text = HEADER.format(sha=sha[:16], gt_ds=GT_DATASET, prep=prep_dataset or PREP_KERNEL, jobs=jobs,
                         iters=iters if mode == "full" else f"10 (smoke; участки: {regions or 'все'})") + text
    meta = json.loads(SRC_META.read_text(encoding="utf-8"))
    meta.update(id=f"{OWNER}/{slug}", title=slug, code_file=f"{slug}.py", is_private=True, enable_gpu=True,
                enable_internet=True, dataset_sources=[GT_DATASET], kernel_sources=[PREP_KERNEL])
    if any(c12):
        meta.update(dataset_sources=[GT_DATASET, prep_dataset], kernel_sources=[])
    return text, meta


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--slug", required=True, help="имя ядра на Kaggle (латиница, дефисы)")
    ap.add_argument("--jobs", required=True, help="например c5-in:0,c5-in:1")
    ap.add_argument("--iters", type=int, default=12000)
    ap.add_argument("--out", type=Path, required=True, help="каталог сборки (вне git)")
    ap.add_argument("--mode", default="full", choices=("full", "smoke"))
    ap.add_argument("--regions", default="", help="только smoke: участки через запятую")
    ap.add_argument("--prep-dataset", default="", help="только c12: приватный датасет cnx_relief (12 каналов)")
    a = ap.parse_args(argv)
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{4,49}", a.slug):
        ap.error("slug: строчная латиница, цифры, дефисы, 5–50 символов")
    text, meta = build(a.slug, a.jobs, a.iters, mode=a.mode, regions=a.regions, prep_dataset=a.prep_dataset)
    d = a.out / a.slug
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{a.slug}.py").write_text(text, encoding="utf-8")
    (d / "kernel-metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"собрано: {d} (jobs {a.jobs}, iters {a.iters}, sha256 ядра "
          f"{hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
