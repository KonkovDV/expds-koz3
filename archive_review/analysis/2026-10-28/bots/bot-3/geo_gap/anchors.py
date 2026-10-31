"""Малые якоря object_poly в полюсе недоступности (своя реализация идеи из COMPETITORS_DEEP §6 п.2)."""
import shapely
from shapely.geometry import box
from shapely.ops import polylabel
def pole(g):
    if g.geom_type == 'MultiPolygon': g = max(g.geoms, key=lambda x: x.area)
    try: pt = polylabel(g, tolerance=0.02)
    except Exception: pt = g.representative_point()
    if not g.contains(pt): pt = g.representative_point()
    return pt, g.exterior.distance(pt) if g.geom_type == 'Polygon' else 0.0
def anchor(p, mode):
    g, conf, reg, grp, mk = p
    pt, d = pole(g)
    if mode == 'H2':
        return (pt.buffer(min(max(0.8 * d, 0.05), 1.0), quad_segs=8), conf, reg, grp, mk)
    if grp.startswith('etgr'):
        h = 0.125; return (box(pt.x - h, pt.y - h, pt.x + h, pt.y + h), conf, reg, grp, mk)
    return (pt.buffer(max(0.5, min(1.0, 0.8 * d)), quad_segs=8), conf, reg, grp, mk)
def apply(preds, mode):
    if mode in ('H1', 'H2'): return [anchor(p, mode) for p in preds]
    if mode == 'H3': return [anchor(p, 'H1') for p in preds] + [(p[0], p[1], p[2], p[3] + '_poly', p[4]) for p in preds]
    return preds
