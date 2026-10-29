"""bot-7: прогон main.py распакованной сдачи с перехватом шагов для проверок ORSP (a)–(e). Ничего не меняет в выходе.
    python run_hooked.py <корень распакованной сдачи> <input_dir> <output.geojson> <dump.json>
Пишет в dump: (c) сырые sp_focal2/sp_or курганы до finish (вкл. sp_partner_only); (b) уверенность до/после
проверки кропа (sp_crop.pt) и до/после слияния; счётчики объектов по (группа, класс) до/после apply_thresholds
и merge_within_markup (d)."""
import collections, json, runpy, sys
from pathlib import Path
ROOT, INP, OUT, DUMP = map(Path, sys.argv[1:5])
sys.path.insert(0, str(ROOT))
from expds import post, pipeline, watchdog   # noqa: E402
KUR = ("kurgany_povrezhdennye", "kurgany_tselye")
D = {"verify": [], "fusion": [], "thresholds": [], "merge": []}

def grp(f):
    p = f["properties"] if isinstance(f.get("properties"), dict) else f
    return p.get("threshold_group") or p.get("markup_type") or ""

def cnt(feats):
    c = collections.Counter((grp(f), f.get("class_name") or (f.get("properties") or {}).get("class_name")) for f in feats)
    return {f"{g}|{k}": n for (g, k), n in sorted(c.items())}

def sp(feats):
    out = []
    for f in feats:
        p = f["properties"] if isinstance(f.get("properties"), dict) else f
        if p.get("class_name") in KUR and grp(f) in ("sp_or", "sp_focal", "sp_focal2"):
            g = f.get("geom"); c = g.centroid if g is not None else None
            out.append({"id": id(f), "region": p.get("region_name"), "group": grp(f), "class": p.get("class_name"),
                        "conf": float(p.get("confidence", 0.0)), "partner_only": bool(p.get("sp_partner_only")),
                        "x": c.x if c is not None else None, "y": c.y if c is not None else None})
    return out

_fus = post.sp_score_fusion
def fus(feats, spec=None):
    before = sp(feats); res = _fus(feats, spec); after = sp(res)
    D["fusion"].append({"before": before, "after": after, "n_in": len(feats), "n_out": len(res)})
    return res
post.sp_score_fusion = fus
_thr = post.apply_thresholds
def thr(feats, *a, **k):
    res = _thr(feats, *a, **k); D["thresholds"].append({"before": cnt(feats), "after": cnt(res), "sp_after": sp(res)}); return res
post.apply_thresholds = thr
_mg = post.merge_within_markup
def mg(feats):
    res = _mg(feats); D["merge"].append({"before": cnt(feats), "after": cnt(res)}); return res
post.merge_within_markup = mg
try:
    from expds import crop_model
    _mv = crop_model.maybe_verify
    def mv(feats, *a, **k):
        b = sp(feats); res = _mv(feats, *a, **k)
        D["verify"].append({"before": b, "after": sp(res) if res is not None else None}); return res
    crop_model.maybe_verify = mv
except Exception as e:  # noqa: BLE001
    D["verify_hook_error"] = repr(e)
sys.argv = [str(ROOT / "inference/main.py"), str(INP), str(OUT)]
try:
    runpy.run_path(str(ROOT / "inference/main.py"), run_name="__main__")
except SystemExit as e:
    D["exit"] = e.code
finally:
    Path(DUMP).write_text(json.dumps(D, ensure_ascii=False), encoding="utf-8")
