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

JOB_RE = re.compile(r"^(c3|c5)-(in|sc):[0-4]$")


def _replace_once(text: str, old: str, new: str, what: str) -> str:
    n = text.count(old)
    if n != 1:
        raise SystemExit(f"правка «{what}»: якорь найден {n} раз(а), нужен ровно один — снимок ядра изменился?")
    return text.replace(old, new)


def build(slug: str, jobs: str, iters: int, src: Path = SRC) -> tuple[str, dict]:
    """Текст ядра и kernel-metadata.json."""
    for tok in jobs.split(","):
        if not JOB_RE.match(tok.strip()):
            raise SystemExit(f"задание {tok!r}: ожидается c3|c5-in|sc:0..4")
    raw = src.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if sha != SRC_SHA256:
        raise SystemExit(f"снимок ядра изменился: sha256 {sha} ≠ {SRC_SHA256}")
    text = raw.decode("utf-8")
    text = _replace_once(text, "CNX_JOBS_DEFAULT = 'c3-sc:4'\n", f"CNX_JOBS_DEFAULT = {jobs!r}\n", "1 задания")
    text = _replace_once(text, 'iters=E("CNX_ITERS", 10 if SMOKE else 12000)',
                         f'iters=E("CNX_ITERS", 10 if SMOKE else {int(iters)})', "2 итерации")
    text = _replace_once(text, S3_OLD, HELPERS + S3_NEW, "3 разметка без S3")
    text = _replace_once(text, INV_OLD, INV_NEW, "4 инвентарь без S3")
    text = _replace_once(text, LOADGT_OLD, LOADGT_NEW, "5 load_gt из файла")
    text = HEADER.format(sha=sha[:16], gt_ds=GT_DATASET, prep=PREP_KERNEL, jobs=jobs, iters=iters) + text
    meta = json.loads(SRC_META.read_text(encoding="utf-8"))
    meta.update(id=f"{OWNER}/{slug}", title=slug, code_file=f"{slug}.py", is_private=True, enable_gpu=True,
                enable_internet=True, dataset_sources=[GT_DATASET], kernel_sources=[PREP_KERNEL])
    return text, meta


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--slug", required=True, help="имя ядра на Kaggle (латиница, дефисы)")
    ap.add_argument("--jobs", required=True, help="например c5-in:0,c5-in:1")
    ap.add_argument("--iters", type=int, default=12000)
    ap.add_argument("--out", type=Path, required=True, help="каталог сборки (вне git)")
    a = ap.parse_args(argv)
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{4,49}", a.slug):
        ap.error("slug: строчная латиница, цифры, дефисы, 5–50 символов")
    text, meta = build(a.slug, a.jobs, a.iters)
    d = a.out / a.slug
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{a.slug}.py").write_text(text, encoding="utf-8")
    (d / "kernel-metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"собрано: {d} (jobs {a.jobs}, iters {a.iters}, sha256 ядра "
          f"{hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
