"""Эквивалентность на всём трейне: expds.orgate.decide_table (код пайплайна) на массивах table.pkl (вход = весь стек A,
как при обучении: gq/пул по всему входу), LAS — из входного каталога scan_root. Сравнение с offline_decisions.json."""
import json, pickle, resource, sys, time, gc
from pathlib import Path
import numpy as np
sys.path.insert(0, "/workspace/koz3_shared/bot-6/repo")
from expds import orgate
from expds.catalog import scan_root
W = Path("/workspace/koz3_shared/bot-6/work/orgate_final")
t0 = time.time()
tab = pickle.load(open("/workspace/rerank/work/table.pkl", "rb"))
T = {k: tab[k] for k in ("cls", "conf", "grp", "mk", "reg", "area", "perim", "cx", "cy", "bw", "bh", "nv")}
del tab; gc.collect()
T["has_las"] = np.zeros(len(T["cls"]), bool)
cat = orgate.catalog_info(scan_root("/workspace/data/train-expds-3"))
# gq и пул в коде R считаются по всему входу; весь стек (1,39 млн строк) в 2 ГБ одним куском не помещается
# (строковые ключи rerank_arch.features) → глобальные gq/пул считаем один раз (те же функции R на всём стеке),
# а участки прогоняем порциями через decide_table, подставляя глобальные gq/пул вместо пересчёта по порции.
from expds import rerank_arch as R
ug, gcode = np.unique(T["grp"], return_inverse=True)
GQ = R.group_quantile(gcode, T["cls"], T["conf"]); POOL = R.pool_mask(gcode, T["cls"], T["conf"])
cur = {}
R.group_quantile = lambda grp, cls, conf: GQ[cur["ix"]]
R.pool_mask = lambda grp, cls, conf: POOL[cur["ix"]]
t1 = time.time()
model = orgate.load({"model": str(W / "orgate_g.txt"), "meta": str(W / "orgate_g.json")})
arch_regs = np.unique(T["reg"][T["cls"] == R.METRIC_CLASSES.index(R.CLASS)])
info = dict(target=[], kept_idx=[], removed_idx=[], las_regions=[], las_source=None)
order = np.argsort(T["reg"], kind="stable"); sreg = T["reg"][order]
lo = np.searchsorted(sreg, arch_regs, "left"); hi = np.searchsorted(sreg, arch_regs, "right")
chunks, curc, n = [], [], 0
for a_, b_ in zip(lo, hi):   # порции по ≤ 120 тыс. строк (участок целиком, не делится)
    if curc and n + (b_ - a_) > 120_000:
        chunks.append(curc); curc, n = [], 0
    curc.append((a_, b_)); n += b_ - a_
chunks.append(curc)
info["max_chunk_rows"] = 0
for ch in chunks:
    ix = np.sort(np.concatenate([order[a_:b_] for a_, b_ in ch]))
    info["max_chunk_rows"] = max(info["max_chunk_rows"], len(ix))
    cur["ix"] = ix
    sub = {key: v[ix] for key, v in T.items()}
    inf = orgate.decide_table(sub, {}, cat, model)
    for key in ("target", "kept_idx", "removed_idx"):
        info[key] += ix[np.asarray(inf[key], int)].tolist()
    info["las_regions"] += inf["las_regions"]; info["las_source"] = inf["las_source"]
t2 = time.time()
off = json.loads((W / "offline_decisions.json").read_text())
reg = T["reg"]; kept = {}; rem = {}
for i in info["kept_idx"]: kept.setdefault(str(reg[i]), []).append(int(i))
for i in info["removed_idx"]: rem.setdefault(str(reg[i]), []).append(int(i))
bad = [r for r, o in off.items() if sorted(kept.get(r, [])) != sorted(o["kept"]) or sorted(rem.get(r, [])) != sorted(o["removed"])]
res = dict(n_target=len(info["target"]), n_kept=len(info["kept_idx"]), n_removed=len(info["removed_idx"]),
           n_regions_offline=len(off), n_regions_mismatch=len(bad), mismatch=bad[:20], las_source=info["las_source"],
           n_las_regions_with_arch=len(info["las_regions"]), n_chunks=len(chunks), max_chunk_rows=info["max_chunk_rows"],
           regions_example={r: dict(kept=sorted(kept.get(r, [])), removed=sorted(rem.get(r, []))) for r in
                            ("008_Герите", "014_Лешки", "031_Новотитаровская", "116_Каменные_выкладки_6")},
           t_load_s=round(t1 - t0, 1), t_decide_s=round(t2 - t1, 1),
           maxrss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
print(json.dumps({k: v for k, v in res.items() if k != "regions_example"}, ensure_ascii=False))
(W / "equiv/equiv_full.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
