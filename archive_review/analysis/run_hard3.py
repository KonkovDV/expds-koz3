"""Новые LAS Гнёздова в кэш и OOF с copy-paste.

Ждёт файл 102, строит тайлы тем же рецептом, что li_g101, склеивает с hard2
и гоняет проверку вне обучения. Балл пишет в hard3.log.
"""
import csv
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(r"C:\Sloi\expds-koz3_v0.2.0\expds-koz3")
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")
W = Path(r"F:\expds3\work")
S = Path(r"C:\Users\Пользователь\AppData\Local\Temp\expds_materials")
LOG = W / "hard3.log"
LAS102 = Path(
    r"F:\expds3\train-expds-3\102_Гнёздово_Заольшанская"
    r"\01_Гнёздово_Заольшанская_La_точки\01_Гнёздово_Заольшанская_La_точки.las"
)
LAS102_SIZE = 164_672_667
PREV = W / "cache" / "li_tin_hard2_t128"
G101 = W / "cache" / "li_g101"
G102 = W / "cache" / "li_g102"
CACHE = W / "cache" / "li_tin_hard3_t128"


def log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(args: list[str]) -> int:
    log(" ".join(args[:6]))
    return subprocess.run(
        [PY, "-X", "utf8", *args],
        cwd=ROOT,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    ).returncode


def link(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def wait_las() -> None:
    while True:
        if LAS102.is_file() and LAS102.stat().st_size == LAS102_SIZE:
            log(f"las102 готов, {LAS102.stat().st_size} байт")
            return
        got = LAS102.stat().st_size if LAS102.is_file() else 0
        log(f"жду las102, сейчас {got} из {LAS102_SIZE}")
        time.sleep(30)


def build_102() -> None:
    if (G102 / "index.csv").exists():
        log("кэш 102 уже есть")
        return
    sys.path.insert(0, str(ROOT))
    from expds.tiles import build_cache

    meta = build_cache(
        r"F:\expds3\train-expds-3",
        r"F:\expds3\work\gt_train.geojson",
        "li",
        str(G102),
        res=0.5,
        tile=128,
        dem={"res": 0.1, "fill": "delaunay"},
        only_regions=["102_Гнёздово_Заольшанская"],
    )
    log(f"тайлы 102: {meta.get('tiles')} положительных {meta.get('positive')}")


def merge() -> None:
    if CACHE.exists():
        shutil.rmtree(CACHE)
    CACHE.mkdir(parents=True)
    for src in PREV.iterdir():
        if src.name != "index.csv":
            link(src, CACHE / src.name)
    extras = [(G101, 9002), (G102, 9003)]
    rows = []
    with (PREV / "index.csv").open(encoding="utf-8") as fh:
        rows.extend(list(csv.reader(fh)))
    for folder, shard in extras:
        for npy in folder.glob("*.npy"):
            link(npy, CACHE / npy.name.replace("_0000.npy", f"_{shard:04d}.npy"))
        with (folder / "index.csv").open(encoding="utf-8") as fh:
            for i, row in enumerate(csv.reader(fh)):
                if i == 0 or not row:
                    continue
                row[0] = str(shard)
                rows.append(row)
    with (CACHE / "index.csv").open("w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(rows)
    log(f"кэш строк {len(rows) - 1}")


def main() -> None:
    LOG.write_text("", encoding="utf-8")
    wait_las()
    build_102()
    merge()
    pred = W / "li_hard3_oof.geojson"
    rc = run([
        "scripts/nn_oof.py", "--markup", "li", "--root", r"F:\expds3\train-expds-3",
        "--cache", str(CACHE), "--work", str(W / "li_hard3_oof"), "--out", str(pred),
        "--folds", "5", "--epochs", "30", "--batch", "32", "--copy-paste", "0.5",
        "--config", str(ROOT / "configs" / "default.toml"),
    ])
    log(f"oof rc={rc}")
    if rc != 0 or not pred.exists():
        raise SystemExit(rc or 1)
    fused = W / "fusion" / "li_hard3_fused.geojson"
    fused.parent.mkdir(parents=True, exist_ok=True)
    rc = run([str(S / "fusion_apply.py"), r"F:\expds3\train-expds-3", str(pred), str(fused)])
    log(f"fusion rc={rc}")
    for name, regions in (("las15", str(W / "regions_las15.txt")),
                          ("new", str(W / "regions_las_new.txt")),
                          ("all", "-")):
        r = subprocess.run(
            [PY, "-X", "utf8", str(S / "sweep_many.py"), str(W / "gt_train.geojson"), regions, "1", str(fused)],
            cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        log(f"sweep {name} {r.stdout.strip()}")
    log("hard3 done")


if __name__ == "__main__":
    main()
