# replace_eval (бот 9): ConvNeXt лидара (li_cnx) как ЗАМЕНА лидарных групп курганов стека бота 10, а не добавка.
# Аудит origin/claude/audit-2026-10-29, docs/AUDIT_CLAUDE_2026-10-29.md §1.3 п.1–2, §6.5. bot-10 — только чтение.
# Входы: все OOF ConvNeXt (по умолчанию /workspace/kaggle_out/koz3-cnx-li-c3*/oof_cnx_*.geojson без *_fN; переопределить CNX_OOF="a b ...").
# Затронутые семьи = site_family участков OOF (+ члены затронутых кластеров family_clusters.json).
#   gt        -> gt_t.geojson (весь GT затронутых семей)
#   fam V     -> fam_V/{семья}.jsonl
#     base      симлинки на bot-10/work/fam (сверка с bot-10/work/curves)
#     add       база + все li_cnx (4 класса) — «добавка», как c3sc-f01/f23 (вариант b)
#     rep_raw   база БЕЗ лидарных курганов (markup li, классы курганов: группы li, li_mosaic, li_dev) на участках, где есть OOF cnx,
#               + li_cnx только курганы (повр., целые); городища/фортификации cnx не добавляются (вариант c)
#     rep_<m>   то же, li_cnx курганы откалиброваны site_calibration (режим m = mix | rank | quantile; floor 0,05, min_n 30, alpha 0,5, q 0,99)
#               — код функции дословно из vendor/post_audit.py (= expds/post.py ветки аудита), вариант d
import sys, json, os, re, glob, ast, collections
from collections import defaultdict
from pathlib import Path
B10 = Path("/workspace/koz3_shared/bot-10"); sys.path.insert(0, str(B10 / "repo"))
from expds.naming import site_family
OUT = Path(__file__).resolve().parent; WD = Path(os.environ.get("RE_WORK", OUT / "work")); WD.mkdir(exist_ok=True); os.chdir(WD)
KUR = ("kurgany_povrezhdennye", "kurgany_tselye")
FAM_SRC = Path(os.environ.get("FAM_SRC", WD / "fam_src"))  # стек бота 10 по семьям (split_stack_by_family.py из /workspace/rerank/stack_arch.geojson); bot-10/work/fam удалён ботом 10 в 08:21
LI_KUR_GROUPS = {None, "", "li", "li_mosaic", "li_dev"}  # группы markup li у курганов в bot-10/work/fam (stack_kurgan_groups.json)
src = os.environ.get("CNX_OOF", "").split() or sorted(p for p in glob.glob("/workspace/kaggle_out/koz3-cnx-li-c3*/oof_cnx_*.geojson") if not re.search(r"_f\d+\.geojson$", p))
oof, seen = [], {}
for p in src:
    fs = json.load(open(p, encoding="utf-8"))["features"]
    for r in {f["properties"]["region_name"] for f in fs}:
        assert r not in seen, f"участок {r} в двух OOF: {seen[r]} и {p}"; seen[r] = p
    oof += fs
CNX_REG = set(seen)
CL = json.load(open(B10 / "work/family_clusters.json", encoding="utf-8"))["clusters"]
FAMS = {site_family(r) for r in CNX_REG}
for k, mem in CL.items():
    if FAMS & set(mem): FAMS |= set(mem)
FAMS = sorted(FAMS)

def calib_fn():
    t = (OUT / "vendor/post_audit.py").read_text(encoding="utf-8"); tree = ast.parse(t); keep = []
    for n in tree.body:
        if isinstance(n, ast.FunctionDef) and n.name in ("site_calibration", "_group_of"): keep.append(n)
        if isinstance(n, ast.Assign) and any(getattr(x, "id", "") in ("KURGANS", "CALIB_MODES") for x in n.targets): keep.append(n)
    ns = {"defaultdict": defaultdict}; exec(compile(ast.Module(body=keep, type_ignores=[]), "post_audit.py", "exec"), ns); return ns["site_calibration"]

cmd = sys.argv[1]
if cmd == "gt":
    gt = json.load(open("/workspace/rerank/gt_train_v2.geojson", encoding="utf-8"))["features"]
    gt_t = [f for f in gt if site_family(f["properties"]["region_name"]) in FAMS]
    json.dump({"type": "FeatureCollection", "features": gt_t}, open("gt_t.geojson", "w", encoding="utf-8"), ensure_ascii=False)
    rf = {}
    for p in src:
        q = Path(p).parent / "region_fold.json"
        if q.exists():
            for r, fo in json.load(open(q))["region_fold"].items():
                if r in CNX_REG: rf[r] = fo
    print(json.dumps({"oof": src, "n_regions": len(CNX_REG), "regions": sorted(CNX_REG), "region_fold": rf, "fams": FAMS, "n_fams": len(FAMS), "gt_t": len(gt_t),
                      "oof_cls": collections.Counter(f["properties"]["class_name"] for f in oof)}, ensure_ascii=False))
    sys.exit()
V = sys.argv[2]; d = WD / f"fam_{V}"; d.mkdir(exist_ok=True)
if V.startswith("rep_"):
    oof = [f for f in oof if f["properties"]["class_name"] in KUR]
rep = {"variant": V}
if V.startswith("rep_") and V != "rep_raw":
    mode = V[4:]; sc = calib_fn()
    flat = [dict(f["properties"]) for f in oof]
    out = sc(flat, {"groups": ["li_cnx"], "classes": list(KUR), "mode": mode, "alpha": 0.5, "q_level": 0.99, "floor": 0.05, "min_n": 30})
    rep["calibrated"] = sum(1 for p in out if "conf_raw" in p); rep["uncalibrated"] = len(out) - rep["calibrated"]
    oof = [{"type": "Feature", "geometry": f["geometry"], "properties": p} for f, p in zip(oof, out)]
extra = defaultdict(list)
for f in oof:
    p = f["properties"]; assert p.get("threshold_group") == "li_cnx"
    extra[site_family(p["region_name"])].append(json.dumps({"type": "Feature", "geometry": f["geometry"], "properties": {k: p[k] for k in ("class_name", "confidence", "region_name", "crs", "markup_type", "threshold_group")}}, ensure_ascii=False))
rep["cnx_feats"] = sum(len(v) for v in extra.values()); removed = collections.Counter(); li_kur_reg_nocnx = collections.Counter()
for m in FAMS:
    srcf = FAM_SRC / f"{m}.jsonl"; assert srcf.exists(), srcf; dst = d / f"{m}.jsonl"
    if dst.exists() or dst.is_symlink(): dst.unlink()
    if V == "base": os.symlink(srcf, dst); continue
    nb = 0
    with open(dst, "w", encoding="utf-8") as fh:
        for l in open(srcf, encoding="utf-8"):
            if V.startswith("rep_"):
                pr = json.loads(l)["properties"]
                if pr.get("markup_type") == "li" and pr["class_name"] in KUR:
                    if pr.get("threshold_group") not in LI_KUR_GROUPS: raise SystemExit(f"неожиданная группа {pr.get('threshold_group')}")
                    if pr["region_name"] in CNX_REG:
                        removed[f"{pr.get('threshold_group') or 'li'}|{pr['class_name']}"] += 1; continue
                    li_kur_reg_nocnx[pr["region_name"]] += 1
            fh.write(l if l.endswith("\n") else l + "\n"); nb += 1
        for l in extra.get(m, []): fh.write(l + "\n")
    rep[m] = {"base_lines_kept": nb, "cnx": len(extra.get(m, []))}
rep["removed_li_kurgan_by_group"] = dict(removed); rep["li_kurgan_kept_regions_without_cnx"] = dict(li_kur_reg_nocnx)
rep["size_MB"] = round(sum(p.stat().st_size for p in d.glob("*.jsonl") if not p.is_symlink()) / 2**20, 1)
json.dump(rep, open(f"fam_{V}.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps({k: v for k, v in rep.items() if k not in FAMS}, ensure_ascii=False))
