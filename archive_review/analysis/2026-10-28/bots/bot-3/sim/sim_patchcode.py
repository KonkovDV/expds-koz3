"""Та же симуляция, но разрезание и полюс считает код патча (expds.geophys.split_merged, label_poles),
а не scikit-image: проверка, что выводы держатся на поставляемом коде."""
import sys, numpy as np
import sim_poly as S
from expds.geophys import GeoParams, split_merged, label_poles
from shapely.geometry import Point

FIELD_THR = 0.5   # порог симуляции; в патче глубина седловины — в долях poly_k

def split_ws(comp, field, signal, h, res):
    p = GeoParams(split=signal, split_h=(h / FIELD_THR if signal == "field" else h), poly_k=FIELD_THR, min_area_m2=res * res)
    lab, n = split_merged(comp.astype(np.int32), np.where(comp, field, 0).astype(np.float32), res, p)
    return [lab == k for k in range(1, n + 1)] or [comp]

def pole(mask, org, res):
    ks, r, c = label_poles(mask.astype(np.int32))
    return Point(org[0] + (c[0] + 0.5) * res, org[1] - (r[0] + 0.5) * res).buffer(S.R_CIRCLE, quad_segs=8)

S.split_ws = split_ws
S.pole_circle_from_mask = pole
S.WS_GRID = [("dt", 0.25), ("dt", 0.5), ("dt", 1.0), ("field", 0.05), ("field", 0.1)]
S.OUT_TAG = "_patchcode"
if __name__ == "__main__":
    s, n = float(sys.argv[1]), float(sys.argv[2])
    S.main(sigmas=(s,), noises=(n,))
