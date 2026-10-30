"""P-W2-4 (Idea 5): snap kurgan centers of li groups to the local lrm_s peak. Default OFF.
usage:
  center_snap.py transform --in stack.geojson --out out.geojson --relief DIR [--center-snap]   (without flag = passthrough copy)
  center_snap.py stage1 --in stack.geojson --gt gt.geojson --relief DIR --out stage1.json
Params fixed by PREREG P-W2-4: R=3 m, sigma=1 px, delta=4 u8. Geometry translated, confidence unchanged."""
import argparse, json, glob, os, sys, numpy as np
from pathlib import Path
from scipy import ndimage as ndi
from pyproj import Transformer
from shapely.geometry import shape, mapping, Point
from shapely.affinity import translate
from shapely.strtree import STRtree
R_M, SIGMA_PX, DELTA = 3.0, 1.0, 4.0
KUR = ('kurgany_povrezhdennye', 'kurgany_tselye')

class Relief:
    def __init__(self, d):
        self.d = Path(d); self.cache = {}; self.tr = {}
    def get(self, region):
        if region not in self.cache:
            j = self.d / f'{region}.json'; n = self.d / f'{region}.npy'
            if not j.exists() or not n.exists(): self.cache[region] = None; return None
            m = json.loads(j.read_text(encoding='utf-8')); a = np.load(n, mmap_mode='r')
            ch = a[m['tags'].index('lrm_s')].astype(np.float32)
            self.cache[region] = (m, ndi.gaussian_filter(ch, SIGMA_PX))
            self.tr[region] = (Transformer.from_crs(3857, m['crs'], always_xy=True), Transformer.from_crs(m['crs'], 3857, always_xy=True))
        return self.cache[region]

def snap_point(rel, region, x3857, y3857):
    """returns (dx3857, dy3857) or None"""
    r = rel.get(region)
    if r is None: return None
    m, L = r; fwd, inv = rel.tr[region]
    ux, uy = fwd.transform(x3857, y3857); st = m['step']
    c = (ux - m['left']) / st; rr = (m['top'] - uy) / st
    ci, ri = int(np.floor(c)), int(np.floor(rr)); k = int(np.ceil(R_M / st))
    H, W = L.shape
    if not (0 <= ri < H and 0 <= ci < W): return None
    r0, r1, c0, c1 = max(ri - k, 0), min(ri + k + 1, H), max(ci - k, 0), min(ci + k + 1, W)
    win = L[r0:r1, c0:c1]
    yy, xx = np.mgrid[r0:r1, c0:c1]; disk = (yy - ri) ** 2 + (xx - ci) ** 2 <= k * k
    w = np.where(disk, win, -np.inf); a = int(np.argmax(w)); pr, pc = np.unravel_index(a, w.shape)
    if w[pr, pc] - L[ri, ci] < DELTA: return None
    nux = m['left'] + (c0 + pc + 0.5) * st; nuy = m['top'] - (r0 + pr + 0.5) * st
    # shift in UTM from the pixel-centre of the original point
    dux, duy = nux - (m['left'] + (ci + 0.5) * st), nuy - (m['top'] - (ri + 0.5) * st)
    nx, ny = inv.transform(ux + dux, uy + duy)
    return nx - x3857, ny - y3857

def eligible(p):
    g = str(p.get('threshold_group') or p.get('markup_type') or '')
    return p.get('class_name') in KUR and 'li' in g

def iter_feats(path):
    import ijson
    with open(path, 'rb') as fh:
        yield from ijson.items(fh, 'features.item', use_float=True)

def cmd_transform(a):
    rel = Relief(a.relief); n = moved = 0
    with open(a.out, 'w', encoding='utf-8') as o:
        o.write('{"type":"FeatureCollection","features":[\n'); first = True
        for f in iter_feats(a.inp):
            n += 1; p = f['properties']
            if a.center_snap and eligible(p):
                g = shape(f['geometry']); rp = g.representative_point()
                d = snap_point(rel, p.get('region_name'), rp.x, rp.y)
                if d is not None: f['geometry'] = mapping(translate(g, d[0], d[1])); moved += 1
            o.write(('' if first else ',\n') + json.dumps(f, ensure_ascii=False)); first = False
        o.write('\n]}\n')
    print(json.dumps(dict(n=n, moved=moved, flag=bool(a.center_snap))))

def cmd_stage1(a):
    rel = Relief(a.relief)
    gt = json.load(open(a.gt, encoding='utf-8'))
    G = {}
    for f in gt['features']:
        p = f['properties']; c = p.get('class_name')
        if c in KUR: G.setdefault((p.get('region_name'), c, p.get('markup_type')), []).append(shape(f['geometry']))
    T = {k: (STRtree(v), v) for k, v in G.items()}
    def inside(region, c, pt):
        t = T.get((region,) + tuple(c))
        if not t: return False
        return any(t[1][i].covers(pt) for i in t[0].query(pt))
    def near(region, c, pt, d):
        t = T.get((region,) + tuple(c))
        if not t: return False
        return any(t[1][i].distance(pt) <= d for i in t[0].query(pt.buffer(d)))
    S = dict(n_elig=0, no_relief=0, moved=0, tp_before=0, G=0, L=0, near5_out=0, near5_moved=0, near5_G=0)
    shifts = []; byconf = []
    for f in iter_feats(a.inp):
        p = f['properties']
        if not eligible(p): continue
        S['n_elig'] += 1; reg, c = p.get('region_name'), (p['class_name'], p.get('markup_type'))
        if rel.get(reg) is None: S['no_relief'] += 1; continue
        g = shape(f['geometry']); rp = g.representative_point()
        b = inside(reg, c, rp); S['tp_before'] += b
        nb = (not b) and near(reg, c, rp, 5.0); S['near5_out'] += nb
        d = snap_point(rel, reg, rp.x, rp.y)
        if d is None: continue
        S['moved'] += 1; S['near5_moved'] += nb; shifts.append(float(np.hypot(*d)))
        rp2 = translate(g, d[0], d[1]).representative_point(); a2 = inside(reg, c, rp2)
        if a2 and not b: S['G'] += 1; S['near5_G'] += nb; byconf.append((float(p.get('confidence', 0)), 1))
        if b and not a2: S['L'] += 1; byconf.append((float(p.get('confidence', 0)), -1))
    S['shift_m_3857_p50'] = float(np.median(shifts)) if shifts else None
    S['net'] = S['G'] - S['L']; S['net_frac_of_tp'] = S['net'] / max(S['tp_before'], 1)
    bc = np.array(byconf) if byconf else np.zeros((0, 2))
    for q in (0.1, 0.3, 0.5):
        m = bc[:, 0] >= q if len(bc) else np.zeros(0, bool)
        S[f'net_conf>={q}'] = int(bc[m, 1].sum()) if len(bc) else 0
    S['stop'] = bool(S['net'] <= 0 or S['net_frac_of_tp'] < 0.005)
    json.dump(S, open(a.out, 'w'), indent=1); print(json.dumps(S))

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest='cmd', required=True)
    t = sp.add_parser('transform'); t.add_argument('--in', dest='inp', required=True); t.add_argument('--out', required=True); t.add_argument('--relief', required=True); t.add_argument('--center-snap', action='store_true')
    s = sp.add_parser('stage1'); s.add_argument('--in', dest='inp', required=True); s.add_argument('--gt', required=True); s.add_argument('--relief', required=True); s.add_argument('--out', required=True)
    a = ap.parse_args(); {'transform': cmd_transform, 'stage1': cmd_stage1}[a.cmd](a)
