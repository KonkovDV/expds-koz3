"""Пороги групп «в выборке» (sweep_groups по всем участкам) для базы и курганов 0,5 м (без микрокругов архитектуры). Проверка: на базе должны
совпасть с configs/default.toml 24bd916 — тогда пороги пакета получены той же процедурой."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "repo"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from expds import metric as M
from honest_bootstrap import load_curves
out = {}
for v in ["base", "kur"]:
    cv = load_curves(Path(f"work2/curves_{v}"))
    res = M.sweep_groups(cv)
    taus = {}
    for cls, r in res.items():
        for g, gv in r["groups"].items(): taus.setdefault(g, {})[cls] = gv["threshold"]
    out[v] = {"taus": taus, "insample": M.score_group_thresholds(cv, taus)[0]}
    print(v, out[v]["insample"], flush=True)
Path("work3/insample_tau_kur.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float))
