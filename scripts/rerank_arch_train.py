"""Финальная модель реранка архитектуры для [rerank_arch]. Только коробка: читает артефакты
/workspace/rerank (стек stack_arch.geojson → work/table.pkl, X2.npy, pool.npy, first.npy, nested_v2.npz).

Модель — как во внешнем фолде nested2/nested3 (те же параметры, 300 раундов, пул, метка first-hit),
но на всех семьях. Пороги групп — sweep_groups в выборке по OOF-баллам канона (nested_v2 outer),
той же процедурой, что пороги базы в configs/default.toml (проверено work3/insample_tau.json).
Пишет <out>.txt (LightGBM) и <out>.json (meta для expds/rerank_arch.py).

    OMP_NUM_THREADS=2 nice -n 10 /workspace/rerank/venv/bin/python scripts/rerank_arch_train.py --out work3/rerank_arch_lgbm
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--rerank", default="/workspace/rerank")
ap.add_argument("--out", required=True)
ap.add_argument("--threads", type=int, default=2)
a = ap.parse_args()
W = Path(a.rerank); sys.path.insert(0, str(W))
import fast  # noqa: E402  (/workspace/rerank/fast.py: точная копия build_curves на готовых парах)
import lightgbm as lgb  # noqa: E402
from expds.metric import sweep_groups  # noqa: E402

import importlib.util  # noqa: E402

# expds уже импортирован из /workspace/expds-koz3 (common.py); модуль ветки грузим по пути
_spec = importlib.util.spec_from_file_location("rerank_arch", Path(__file__).resolve().parents[1] / "expds/rerank_arch.py")
R = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(R)

T = fast.T; cls = T["cls"].astype(int); conf = T["conf"]
assert tuple(fast.CL) == R.METRIC_CLASSES
names = json.load(open(W / "work/X2_names.json")); assert names == R.NAMES
X = np.load(W / "work/X2.npy", mmap_mode="r"); pool = np.load(W / "work/pool.npy"); y = np.load(W / "work/first.npy")
A = R.METRIC_CLASSES.index(R.CLASS)
tr = (cls == A) & pool
mono = [1 if n in ("conf", "logit") else 0 for n in names]; cat = [names.index(n) for n in ("cls", "grp", "mk")]
P = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=100, feature_fraction=0.8,
         bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, monotone_constraints=mono, verbose=-1,
         num_threads=a.threads, seed=0)
booster = lgb.train(P, lgb.Dataset(np.asarray(X[tr]), y[tr].astype(float), categorical_feature=cat), 300)
out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
booster.save_model(str(out.with_suffix(".txt")))
# пороги: баллы архитектуры = OOF канона (-1 вне пула), остальные классы — уверенность стека
outer = np.load(W / "work/nested_v2.npz")["outer"]
S = conf.copy(); S[cls == A] = outer[cls == A]
cv = fast.curves(score=S)
res = sweep_groups({R.CLASS: cv[R.CLASS]})
taus = {g: float(gv["threshold"]) for g, gv in res[R.CLASS]["groups"].items()}
base = sweep_groups({R.CLASS: fast.curves()[R.CLASS]})
meta = dict(names=names, classes=list(R.METRIC_CLASSES), grp_vocab=sorted(set(T["grp"].astype(str))),
            mk_vocab=sorted(set(T["mk"].astype(str))), thresholds=taus,
            # в сдаче спутниковая архитектура идёт без threshold_group (группа sp_or), в стеке — sp_arch
            # (configs/default.toml, комментарий у [thresholds_by_markup.sp_or] arkhitektury)
            arch_group_alias={"sp_or": "sp_arch"},
            train=dict(rows=int(tr.sum()), positives=int(y[tr].sum()), rounds=300, params={k: v for k, v in P.items()},
                       stack_sha256="b412e29934dd292bcc343627690cd246fd9048948a85e8a00181c86417f8e253"),
            base_taus={g: float(gv["threshold"]) for g, gv in base[R.CLASS]["groups"].items()})
out.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
print("rows", meta["train"]["rows"], "pos", meta["train"]["positives"], "taus", taus, "base", meta["base_taus"])
