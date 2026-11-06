#!/usr/bin/env python3
"""Задача 16 (bigobj): кривые стека A с преобразованием F1/F2 ветки bigobj.

Как repo/scripts/family_curves.py (семьи + кластеры, пачки, M.match_collected), но объекты
классов из --classes перед сопоставлением проходят функции expds/post.py ветки (импорт из --code):
  F1 big_area_groups(spec)  — threshold_group → <g>_big у прогнозов ≥ min_area_m2 (геометрия та же);
  F2 shape_cores({cls: rule}) — круг-ядро в полюсе недоступности у полигонов ≥ min_area_m2.
Порядок как в watchdog.finish: F1 до порогов, F2 после (геометрия и группа от порога не зависят,
поэтому на стеке без порогов порядок не важен). Склейка merge_within_markup не моделируется
(на A ≈ 0, REPORT2 п. 6). Только чтение входов; пишет *.pkl в --out.
"""
from __future__ import annotations
import argparse, gc, importlib.util, json, pickle, sys, time, types
from pathlib import Path

B10 = Path("/workspace/koz3_shared/bot-10")
sys.path.insert(0, str(B10 / "repo"))
from expds import metric as M  # noqa
from expds.naming import site_family  # noqa
M._SHM_CACHE_DIR = Path("/proc/expds_no_cache")


def load_branch_post(code: Path):
    pkg = types.ModuleType("bigobj_expds"); pkg.__path__ = [str(code / "expds")]
    sys.modules["bigobj_expds"] = pkg
    spec = importlib.util.spec_from_file_location("bigobj_expds.post", code / "expds/post.py")
    mod = importlib.util.module_from_spec(spec); sys.modules["bigobj_expds.post"] = mod
    spec.loader.exec_module(mod)
    return mod


def make_transform(code: Path, variant: dict):
    from shapely.geometry import mapping, shape
    P = load_branch_post(code)
    big, core = variant.get("big_groups"), variant.get("core")
    classes = set((big or {}).get("classes") or ()) | set(core or {})
    keys = [f'"{c}"' for c in classes]

    def tf(f: dict) -> list[dict]:
        p = f.get("properties") or {}
        if p.get("class_name") not in classes:
            return [f]
        feat = dict(p, geom=shape(f["geometry"]))
        out = [feat]
        if big:
            out = P.big_area_groups(out, big)
        if core:
            out = P.shape_cores(out, core)
        res = []
        for x in out:
            g = x.pop("geom")
            res.append({"type": "Feature", "geometry": mapping(g), "properties": x})
        return res
    tf.keys = keys
    return tf


def iter_unit(fam_dir: Path, members, tf):
    for m in members:
        src = fam_dir / f"{m}.jsonl"
        if not src.exists():
            continue
        with open(src, encoding="utf-8") as fh:
            for line in fh:
                if tf is not None and any(k in line for k in tf.keys):
                    yield from tf(json.loads(line))
                else:
                    yield json.loads(line)


def units_of(fams, clusters):
    units = {f: [f] for f in fams}
    for members in json.loads(Path(clusters).read_text(encoding="utf-8"))["clusters"].values():
        for m in members:
            units.pop(m, None)
        units["+".join(sorted(members))] = sorted(members)
    return units


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fam", default=str(B10 / "work11/fam0"))
    ap.add_argument("--gt", default="/workspace/rerank/gt_train_v2.geojson")
    ap.add_argument("--clusters", default=str(B10 / "work/family_clusters.json"))
    ap.add_argument("--code", required=True, help="корень кода ветки bot-2/bigobj (expds/post.py)")
    ap.add_argument("--variant", required=True, help="JSON: {big_groups: {...}, core: {класс: {...}}}")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=20000)
    a = ap.parse_args()
    variant = json.loads(a.variant)
    tf = make_transform(Path(a.code), variant) if variant else None
    mod = M.load_official(None); calc = mod.F1Calculator()
    gt_by_fam = {}
    for f in json.loads(Path(a.gt).read_text(encoding="utf-8"))["features"]:
        reg = str((f.get("properties") or {}).get("region_name", "default_source"))
        gt_by_fam.setdefault(site_family(reg), []).append(f)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "variant.json").write_text(json.dumps(variant, ensure_ascii=False))
    fams = sorted(set(gt_by_fam) | {p.stem for p in Path(a.fam).glob("*.jsonl")})
    sys.path.insert(0, str(B10 / "repo/scripts"))
    from family_curves import collect_batched  # noqa
    for fam, members in sorted(units_of(fams, a.clusters).items()):
        dst = out / f"{fam}.pkl"
        if dst.exists():
            continue
        t0 = time.time()
        preds = collect_batched(mod, calc, iter_unit(Path(a.fam), members, tf), False, a.batch)
        gts = collect_batched(mod, calc, iter([f for m in members for f in gt_by_fam.get(m, [])]), True, a.batch)
        curves = M.match_collected(calc, preds, gts)
        del preds, gts
        tmp = dst.with_suffix(".tmp"); tmp.write_bytes(pickle.dumps(curves, protocol=pickle.HIGHEST_PROTOCOL)); tmp.replace(dst)
        del curves; gc.collect()
        print(f"{fam}\t{time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
