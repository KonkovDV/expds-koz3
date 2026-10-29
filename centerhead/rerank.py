"""Метки кандидатов по официальному правилу матчинга + LightGBM-реранкер OOF по семьям.

Правило (bot-10/repo/materials/compute_metrics.py, koz3-1.5.0):
  полигон-GT: cov = area(P∩G)/min(area) ≥ 0.5 (_polygon_coverage, стр. 364–376) И representative_point(P) ∈ G
  (_inner_point_in_mask, стр. 393–410; _match_polygon_gt, стр. 923–928);
  точка-GT: расстояние центроида P до точки ≤ 2 м (_match_point_gt, стр. 930–944);
  жадно по убыванию confidence, один-к-одному, по классу (_soft_match, стр. 850–907).
Здесь P — круг радиуса circle_r_m (32 вершины, как _point_to_circle стр. 476) в РОДНОЙ метрической СК участка
(метрика считает в UTM после 3857→UTM; для участков в UTM/МСК разница масштаба пренебрежима — не проверялось).
"""
from __future__ import annotations

import math

import numpy as np

from .config import DEFAULTS

COVERAGE_THRESHOLD = 0.5      # compute_metrics.py:184
POINT_R_TOLERANCE = 2.0       # compute_metrics.py:185


def circle(x: float, y: float, r: float, n: int = 32):
    from shapely.geometry import Polygon
    a = 2 * math.pi * np.arange(n) / n
    return Polygon(np.column_stack([x + r * np.cos(a), y + r * np.sin(a)]))


def official_hit(pred, gt) -> bool:
    if gt.geom_type == "Point":
        return pred.centroid.distance(gt) <= POINT_R_TOLERANCE
    inter = pred.intersection(gt).area
    m = min(pred.area, gt.area)
    if m <= 0 or inter / m < COVERAGE_THRESHOLD:
        return False
    return gt.covers(pred.representative_point())


def match_labels(t: dict, gt_geoms: list, gt_cls: np.ndarray, circle_r_m: float = DEFAULTS["circle_r_m"],
                 mode: str = "official") -> tuple[np.ndarray, np.ndarray]:
    """t — таблица кандидатов ОДНОГО участка (X, Y, cls, score); gt_geoms — shapely в той же СК; gt_cls — индексы
    классов (0 tselye, 1 povr). mode="official": жадно по score, один-к-одному (как _soft_match);
    mode="any": метка 1, если кандидат попадает в любой GT своего класса (без конкуренции).
    → (y int8 (N,), matched_gt int64 (N,), −1 = нет)."""
    from shapely import STRtree
    X, Y, s, c = (np.asarray(t[k]) for k in ("X", "Y", "score", "cls"))
    n = len(X)
    y = np.zeros(n, np.int8)
    mg = np.full(n, -1, np.int64)
    if n == 0 or len(gt_geoms) == 0:
        return y, mg
    gt_cls = np.asarray(gt_cls)
    tree = STRtree(list(gt_geoms))
    used = np.zeros(len(gt_geoms), bool)
    for i in np.lexsort((np.arange(n), -s)):
        P = circle(X[i], Y[i], circle_r_m)
        cand = tree.query(P.buffer(POINT_R_TOLERANCE))
        best, best_cov = -1, -math.inf
        for j in cand:
            if gt_cls[j] != c[i] or (mode == "official" and used[j]):
                continue
            g = gt_geoms[j]
            if official_hit(P, g):
                cov = (-P.centroid.distance(g) if g.geom_type == "Point"          # как _match_point_gt: score = −distance
                       else P.intersection(g).area / min(P.area, g.area))
                if cov > best_cov:
                    best, best_cov = j, cov
        if best >= 0:
            y[i] = 1
            mg[i] = best
            if mode == "official":
                used[best] = True
    return y, mg


# ── OOF-реранкер ─────────────────────────────────────────────────────────────────────────────────────────────
LGB_PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=15, min_data_in_leaf=20, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, num_threads=2, deterministic=True,
                  force_row_wise=True, seed=0, verbose=-1)   # нетюнингованные стартовые значения
NUM_ROUNDS = 300


def folds_to_map(folds) -> dict:
    """list[set[region]] (expds.metric.family_folds) | dict region→fold | bot-9 FOLDS["family"]["folds"]
    (list of {"fold", "test_families"/"test_regions"…}) → dict region→fold."""
    if isinstance(folds, dict):
        return dict(folds)
    out = {}
    for k, f in enumerate(folds):
        regs = f if isinstance(f, (set, frozenset, list, tuple)) else f.get("test_regions")
        if regs is None:
            raise ValueError("для формата bot-9 передайте dict region→fold (в JSON фолды заданы семьями)")
        for r in regs:
            if r in out:
                raise ValueError(f"участок {r} в двух фолдах")
            out[r] = k
    return out


def check_no_leak(region_fold: dict, family_of=None, cnn_fold_of: dict | None = None) -> None:
    """Семья не делится между фолдами; если дан cnn_fold_of (region→фолд CNN, где участок был тестом), разбиение
    реранкера должно совпадать с разбиением CNN (иначе признаки train-строк могли видеть тест-фолд)."""
    if family_of is not None:
        fam = {}
        for r, k in region_fold.items():
            f = family_of(r)
            if fam.setdefault(f, k) != k:
                raise ValueError(f"семья {f} в фолдах {fam[f]} и {k}")
    if cnn_fold_of is not None:
        miss = [r for r in region_fold if r not in cnn_fold_of]
        if miss:
            raise ValueError(f"нет OOF-фолда CNN для {miss[:5]} — карты должны быть OOF")
        pairs = {}
        for r, k in region_fold.items():
            if pairs.setdefault(cnn_fold_of[r], k) != k:
                raise ValueError("разбиение реранкера не совпадает с разбиением CNN (фолд CNN "
                                 f"{cnn_fold_of[r]} ↦ фолды реранкера {pairs[cnn_fold_of[r]]} и {k})")


class OOFReranker:
    """fit_oof → OOF-score для каждого кандидата (модель фолда k не видит участков фолда k);
    fit_full → модель на всём для теста; predict → score ∈ [0,1]."""

    def __init__(self, params: dict | None = None, num_rounds: int = NUM_ROUNDS, feature_names=None):
        self.params = dict(LGB_PARAMS, **(params or {}))
        self.num_rounds = num_rounds
        self.feature_names = list(feature_names) if feature_names is not None else None
        self.fold_models: dict[int, object] = {}
        self.full_model = None

    def _train(self, X, y):
        import lightgbm as lgb
        if len(np.unique(y)) < 2:
            return float(np.mean(y)) if len(y) else 0.0     # вырожденный фолд: константа
        ds = lgb.Dataset(np.asarray(X, np.float32), label=np.asarray(y, np.float32),
                         feature_name=self.feature_names or "auto", free_raw_data=True)
        return lgb.train(self.params, ds, num_boost_round=self.num_rounds)

    @staticmethod
    def _pred(m, X):
        if isinstance(m, float):
            return np.full(len(X), m)
        return m.predict(np.asarray(X, np.float32), num_threads=2)

    def fit_oof(self, X, y, regions, folds, family_of=None, cnn_fold_of=None) -> np.ndarray:
        regions = np.asarray(regions)
        rf = folds_to_map(folds)
        miss = sorted(set(regions) - set(rf))
        if miss:
            raise ValueError(f"участки без фолда: {miss[:5]}")
        check_no_leak({r: rf[r] for r in set(regions)}, family_of, cnn_fold_of)
        fold = np.array([rf[r] for r in regions])
        oof = np.full(len(y), np.nan)
        for k in np.unique(fold):
            te = fold == k
            m = self._train(X[~te], y[~te])
            self.fold_models[int(k)] = m
            oof[te] = self._pred(m, X[te])
        self.oof_fold_ = fold
        return oof

    def fit_full(self, X, y):
        self.full_model = self._train(X, y)
        return self

    def predict(self, X, fold: int | None = None) -> np.ndarray:
        """fold=None: полная модель (fit_full), иначе среднее фолд-моделей; fold=k — модель фолда k."""
        if fold is not None:
            return self._pred(self.fold_models[fold], X)
        if self.full_model is not None:
            return self._pred(self.full_model, X)
        if not self.fold_models:
            raise RuntimeError("не обучен")
        return np.mean([self._pred(m, X) for m in self.fold_models.values()], 0)

    def save(self, path) -> None:
        import json
        d = {"params": self.params, "num_rounds": self.num_rounds, "feature_names": self.feature_names,
             "full": _dump(self.full_model), "folds": {str(k): _dump(m) for k, m in self.fold_models.items()}}
        with open(path, "w") as fh:
            json.dump(d, fh)

    @classmethod
    def load(cls, path) -> "OOFReranker":
        import json
        d = json.load(open(path))
        o = cls(d["params"], d["num_rounds"], d["feature_names"])
        o.full_model = _load(d["full"])
        o.fold_models = {int(k): _load(v) for k, v in d["folds"].items()}
        return o


def _dump(m):
    if m is None or isinstance(m, float):
        return m
    return {"lgb": m.model_to_string()}


def _load(v):
    if v is None or isinstance(v, float):
        return v
    import lightgbm as lgb
    return lgb.Booster(model_str=v["lgb"])
