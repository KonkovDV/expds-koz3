# Потоковая проверка OOF D-FINE (без загрузки всего файла в json): CRS, типы, кольца, валидность, площади, классы, счёт по фолдам/участкам.
import json, sys, collections, math, resource
from shapely.geometry import shape
SRC = "/workspace/kaggle_out/koz3-dfine-spor-full/"
def stream(path):
    s = open(path, encoding="utf-8").read()
    head_end = s.index('"features": [') + len('"features": [')
    head = s[:head_end]
    dec = json.JSONDecoder(); i = head_end; n = len(s)
    while True:
        while i < n and s[i] in " \n\r\t,": i += 1
        if s[i] == "]": break
        obj, i = dec.raw_decode(s, i); yield head, obj
    tail = s[i:]
    yield ("TAIL", tail), None
res = {}
for name in ["oof_dfine_spor.geojson", "oof_dfine_f4.geojson", "oof_dfine_f2.geojson", "oof_dfine_f1.geojson"]:
    st = dict(n=0, types=collections.Counter(), props_keys=collections.Counter(), cls=collections.Counter(), fold=collections.Counter(),
              region=collections.Counter(), markup=collections.Counter(), group=collections.Counter(), crs_prop=collections.Counter(),
              empty=0, invalid=0, zero_area=0, open_ring=0, few_pts=0, nonfinite=0, conf_bad=0, n_pts=collections.Counter())
    conf = []; area = []; head = None; fids = set(); dup_fid = 0
    fold_cls = collections.defaultdict(collections.Counter)
    for h, f in stream(SRC + name):
        if f is None: st["tail"] = h[1].strip()[:80]; continue
        head = h; st["n"] += 1
        g = f.get("geometry") or {}; p = f.get("properties") or {}
        st["types"][g.get("type")] += 1
        st["props_keys"][tuple(sorted(p))] += 1
        st["cls"][p.get("class_name")] += 1; st["fold"][p.get("fold")] += 1; st["region"][p.get("region_name")] += 1
        st["markup"][p.get("markup_type")] += 1; st["group"][p.get("group")] += 1; st["crs_prop"][p.get("crs")] += 1
        fold_cls[p.get("fold")][p.get("class_name")] += 1
        fid = (p.get("fold"), p.get("fid"))
        if fid in fids: dup_fid += 1
        fids.add(fid)
        c = p.get("confidence")
        if not isinstance(c, (int, float)) or not (0 <= c <= 1): st["conf_bad"] += 1
        else: conf.append(c)
        co = g.get("coordinates") or []
        if not co or not co[0]: st["empty"] += 1; continue
        ring = co[0]; st["n_pts"][len(ring)] += 1
        if len(ring) < 4: st["few_pts"] += 1
        if ring[0] != ring[-1]: st["open_ring"] += 1
        if not all(math.isfinite(v) for pt in ring for v in pt): st["nonfinite"] += 1; continue
        sh = shape(g)
        if sh.is_empty: st["empty"] += 1
        elif not sh.is_valid: st["invalid"] += 1
        a = sh.area
        if a <= 0: st["zero_area"] += 1
        area.append(a)
    conf.sort(); area.sort()
    q = lambda v, x: round(v[min(len(v) - 1, int(x * len(v)))], 4) if v else None
    out = {k: (dict(v) if isinstance(v, collections.Counter) else v) for k, v in st.items()}
    out["props_keys"] = {"|".join(k): v for k, v in st["props_keys"].items()}
    out["head"] = head[:200]; out["dup_fold_fid"] = dup_fid
    out["conf_q"] = {x: q(conf, x) for x in (0, 0.25, 0.5, 0.75, 0.9, 0.99, 1)}
    out["conf_ge"] = {t: sum(c >= t for c in conf) for t in (0.1, 0.2, 0.3, 0.5, 0.7)}
    out["area_m2_3857_q"] = {x: q(area, x) for x in (0, 0.01, 0.5, 0.99, 1)}
    out["fold_cls"] = {str(k): dict(v) for k, v in fold_cls.items()}
    out["n_regions"] = len(st["region"])
    res[name] = out
    print(name, "n", st["n"], "types", dict(st["types"]), "folds", dict(st["fold"]), "regions", len(st["region"]), "empty", st["empty"], "invalid", st["invalid"],
          "zero_area", st["zero_area"], "open", st["open_ring"], "few", st["few_pts"], "nonfinite", st["nonfinite"], "conf_bad", st["conf_bad"], "dup", dup_fid,
          "maxrss_MB", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024, flush=True)
json.dump(res, open("validate_oof.json", "w"), ensure_ascii=False, indent=1)
