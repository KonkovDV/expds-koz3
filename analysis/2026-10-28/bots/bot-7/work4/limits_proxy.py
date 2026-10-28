"""Прокси-проверка лимитов выдачи кандидата: watchdog.finish (полный конфиг кандидата) + writer.write_geojson
на каждой семье стека OOF до порогов (bot-10/work/fam, 104 семьи, 1,39 млн кандидатов), затем на объединении
крупнейших семей. Считает объекты, вершины, МБ, время и пик RSS. Запуск из bot-7/repo.

    python ../work4/limits_proxy.py [--combined N]
"""
import argparse, json, resource, sys, time, tomllib
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
from shapely.geometry import shape
from expds import watchdog, writer, post

FAM = Path("/workspace/koz3_shared/bot-10/work/fam")
OUT = Path(__file__).resolve().parent
ap = argparse.ArgumentParser(); ap.add_argument("--combined", type=int, default=6); a = ap.parse_args()
cfg = tomllib.loads(Path("configs/default.toml").read_text(encoding="utf-8"))
tmp = OUT / "tmp_result"; tmp.mkdir(exist_ok=True)


def load(fam):
    feats = []
    for line in open(FAM / f"{fam}.jsonl", encoding="utf-8"):
        d = json.loads(line); p = d["properties"]
        f = {k: v for k, v in p.items() if k not in ("fid", "crs")}
        f["geom"] = shape(d["geometry"]); feats.append(f)
    return feats


def run(name, feats):
    t = time.monotonic(); kept = watchdog.finish(feats, cfg); t_fin = time.monotonic() - t
    p = writer.write_geojson(kept, tmp / "result.geojson"); t_all = time.monotonic() - t
    v = writer.validate(p)
    fc = json.loads(p.read_text(encoding="utf-8"))
    nv = [sum(len(r) for r in f["geometry"]["coordinates"]) for f in fc["features"]]
    row = {"name": name, "rows_in": len(feats), "regions": len({f.get("region_name") for f in feats}),
           "after_finish": len(kept), "written": len(fc["features"]), "max_vertices": max(nv, default=0),
           "mb": round(p.stat().st_size / 2**20, 3), "finish_s": round(t_fin, 2), "finish_write_s": round(t_all, 2),
           "validate_ok": v["ok"], "errors": v["errors"][:3], "by_class": dict(Counter(f["properties"]["class_name"] for f in fc["features"])),
           "maxrss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024}
    print(json.dumps(row, ensure_ascii=False), flush=True)
    return row


fams = sorted(p.stem for p in FAM.glob("*.jsonl"))
rows = []
for fam in fams:
    rows.append(run(fam, load(fam)))
(OUT / "limits_per_family.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
big = sorted(rows, key=lambda r: -r["rows_in"])[:a.combined]
feats = [f for r in big for f in load(r["name"])]
comb = run("+".join(r["name"] for r in big), feats)
summ = {"families": len(rows), "rows_in_total": sum(r["rows_in"] for r in rows),
        "written_total_all_families": sum(r["written"] for r in rows),
        "max_written_family": max(rows, key=lambda r: r["written"])["name"], "max_written": max(r["written"] for r in rows),
        "max_vertices": max(r["max_vertices"] for r in rows + [comb]), "all_validate_ok": all(r["validate_ok"] for r in rows + [comb]),
        "max_finish_write_s_family": max(r["finish_write_s"] for r in rows), "sum_finish_write_s": round(sum(r["finish_write_s"] for r in rows), 1),
        "combined": comb, "maxrss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024}
(OUT / "limits_summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1), encoding="utf-8")
print("SUMMARY", json.dumps(summ, ensure_ascii=False))
