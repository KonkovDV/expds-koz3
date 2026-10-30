# koz3-w2-ground-clip-audit (CI2-3 v2, W2). Own code, numpy-only LAS reader (no laspy/internet needed).
# Per LAS: full class histogram; DEM (cell mean z) from class 2 only vs all classes (excl. 7/18 noise) at R m;
# diff stats overall and inside GT polygons (kurgany povr/tselye, gorodishcha): mean/p95 dz, mound height h=p95(in)-median(annulus).
# Also "pipeline" = what expds.lidar does: class 2 if the file has any class-2 point, else all points.
import os, sys, json, glob, struct, time, math
import numpy as np
R = float(os.environ.get("GA_RES", "0.25"))
OUT = "/kaggle/working/gaudit"; os.makedirs(OUT, exist_ok=True)
NOISE = (7, 18)

def las_open(p):
    with open(p, "rb") as h: hd = h.read(375)
    vmin = hd[25]; off = struct.unpack("<I", hd[96:100])[0]; fmt = hd[104] & 0x3f
    rl = struct.unpack("<H", hd[105:107])[0]; n = struct.unpack("<I", hd[107:111])[0]
    if vmin >= 4 and n == 0: n = struct.unpack("<Q", hd[247:255])[0]
    sc = struct.unpack("<3d", hd[131:155]); of = struct.unpack("<3d", hd[155:179])
    mx = struct.unpack("<6d", hd[179:227])  # maxX,minX,maxY,minY,maxZ,minZ
    m = np.memmap(p, dtype=np.uint8, mode="r", offset=off, shape=(n, rl))
    return m, fmt, n, sc, of, (mx[1], mx[3], mx[0], mx[2])

def chunks(m, fmt, sc, of, step=8_000_000):
    for s in range(0, m.shape[0], step):
        b = np.asarray(m[s:s + step])
        xyz = b[:, :12].copy().view("<i4").reshape(-1, 3).astype(np.float64)
        x = xyz[:, 0] * sc[0] + of[0]; y = xyz[:, 1] * sc[1] + of[1]; z = xyz[:, 2] * sc[2] + of[2]
        c = (b[:, 15] & 0x1f) if fmt < 6 else b[:, 16]
        yield x, y, z, c.astype(np.int64)

def site_gt(site):
    try:
        from pyproj import Transformer
    except Exception:
        return []
    epsg = int(json.load(open(f"{site}_UTM.json"))["crs"].split(":")[-1])
    tr = Transformer.from_crs(3857, epsg, always_xy=True)
    P = []
    for gj in sorted(glob.glob(f"{site}_Li_*.geojson")):
        nm = os.path.basename(gj)
        cls = next((c for c in ("tselye", "povr", "gorod") if f"_Li_{c}" in nm), None)
        if cls is None: continue
        for f in json.load(open(gj, encoding="utf-8"))["features"]:
            g = f.get("geometry") or {}
            polys = [g["coordinates"]] if g.get("type") == "Polygon" else g.get("coordinates", []) if g.get("type") == "MultiPolygon" else []
            for pc in polys:
                ring = np.asarray(pc[0], np.float64)
                X, Y = tr.transform(ring[:, 0], ring[:, 1]); P.append((cls, np.c_[X, Y]))
    return P

def audit(site, las):
    t0 = time.time()
    m, fmt, n, sc, of, (x0, y0, x1, y1) = las_open(las)
    W = int((x1 - x0) / R) + 1; H = int((y1 - y0) / R) + 1; N = W * H
    hist = np.zeros(256, np.int64)
    sg = np.zeros(N); cg = np.zeros(N, np.int32); sa = np.zeros(N); ca = np.zeros(N, np.int32)
    for x, y, z, c in chunks(m, fmt, sc, of):
        hist += np.bincount(c, minlength=256)
        col = np.clip(((x - x0) / R).astype(np.int64), 0, W - 1); row = np.clip(((y - y0) / R).astype(np.int64), 0, H - 1)
        lin = row * W + col
        a = ~np.isin(c, NOISE); g = c == 2
        sa += np.bincount(lin[a], weights=z[a], minlength=N); ca += np.bincount(lin[a], minlength=N).astype(np.int32)
        sg += np.bincount(lin[g], weights=z[g], minlength=N); cg += np.bincount(lin[g], minlength=N).astype(np.int32)
    DA = np.where(ca > 0, sa / np.maximum(ca, 1), np.nan).reshape(H, W)
    DG = np.where(cg > 0, sg / np.maximum(cg, 1), np.nan).reshape(H, W)
    del sa, sg
    has2 = bool(hist[2] > 0)
    rep = {"site": os.path.basename(site), "las": os.path.basename(las), "fmt": fmt, "n": int(n), "res": R,
           "classes": {str(k): int(v) for k, v in enumerate(hist) if v},
           "frac": {str(k): round(float(v / max(n, 1)), 4) for k, v in enumerate(hist) if v},
           "frac_ground": round(float(hist[2] / max(n, 1)), 4), "pipeline_uses": "class2" if has2 else "all_points"}
    va = np.isfinite(DA); vg = np.isfinite(DG); both = va & vg
    d = (DA - DG)[both]
    rep["cells_all"] = int(va.sum()); rep["cells_g2"] = int(vg.sum())
    rep["g2_hole_frac_of_all_cells"] = round(float((va & ~vg).sum() / max(va.sum(), 1)), 4)
    if d.size:
        rep["dz_all_minus_g2"] = {"mean": round(float(d.mean()), 3), "p50": round(float(np.median(d)), 3),
                                 "p90": round(float(np.percentile(d, 90)), 3), "p99": round(float(np.percentile(d, 99)), 3),
                                 "frac_gt_0.5m": round(float((d > 0.5).mean()), 4), "frac_gt_2m": round(float((d > 2).mean()), 4)}
    def pip(ring, px, py):
        inside = np.zeros(px.shape, bool); xs, ys = ring[:, 0], ring[:, 1]; j = len(xs) - 1
        for i in range(len(xs)):
            c = ((ys[i] > py) != (ys[j] > py)) & (px < (xs[j] - xs[i]) * (py - ys[i]) / (ys[j] - ys[i] + 1e-30) + xs[i])
            inside ^= c; j = i
        return inside
    rows = []
    for cls, ring in site_gt(site):
        try:
            area = 0.5 * abs(np.dot(ring[:, 0], np.roll(ring[:, 1], 1)) - np.dot(ring[:, 1], np.roll(ring[:, 0], 1)))
            rr = max(3.0, 0.75 * math.sqrt(area / math.pi))
            bx0, by0 = ring.min(0) - rr; bx1, by1 = ring.max(0) + rr
            c0, c1 = max(int((bx0 - x0) / R), 0), min(int((bx1 - x0) / R) + 1, W)
            r0, r1 = max(int((by0 - y0) / R), 0), min(int((by1 - y0) / R) + 1, H)
            if c1 <= c0 or r1 <= r0: continue
            cx = x0 + (np.arange(c0, c1) + 0.5) * R; cy = y0 + (np.arange(r0, r1) + 0.5) * R
            gx, gy = np.meshgrid(cx, cy)
            pin = pip(ring, gx, gy)
            cen = ring.mean(0); dist = np.hypot(gx - cen[0], gy - cen[1]); req = math.sqrt(area / math.pi)
            pann = (~pin) & (dist <= req + rr) & (dist > req + 0.5)
            if pin.sum() < 2 or pann.sum() < 4: continue
            r = {"cls": cls, "area": round(area, 1), "n_in": int(pin.sum())}
            for k, D in (("g2", DG), ("all", DA)):
                w = D[r0:r1, c0:c1]; a = w[pin]; b = w[pann]
                r[f"hole_in_{k}"] = round(float(np.isnan(a).mean()), 3)
                if np.isfinite(a).sum() >= 2 and np.isfinite(b).sum() >= 3:
                    r[f"h_{k}"] = round(float(np.nanpercentile(a, 95) - np.nanmedian(b)), 3)
            wa = DA[r0:r1, c0:c1][pin]; wg = DG[r0:r1, c0:c1][pin]; ok = np.isfinite(wa) & np.isfinite(wg)
            if ok.any(): r["dz_in_mean"] = round(float((wa - wg)[ok].mean()), 3); r["dz_in_max"] = round(float((wa - wg)[ok].max()), 3)
            rows.append(r)
        except Exception as e:
            rows.append({"cls": cls, "err": repr(e)[:200]})
    rep["n_gt"] = len(rows)
    for cls in ("povr", "tselye", "gorod"):
        rr_ = [r for r in rows if r.get("cls") == cls and "h_g2" in r and "h_all" in r]
        if rr_:
            hg = np.array([r["h_g2"] for r in rr_]); ha = np.array([r["h_all"] for r in rr_])
            rep[f"gt_{cls}"] = {"n": len(rr_), "h_g2_med": round(float(np.median(hg)), 3), "h_all_med": round(float(np.median(ha)), 3),
                                "dh_med": round(float(np.median(ha - hg)), 3), "dh_p90": round(float(np.percentile(ha - hg, 90)), 3),
                                "hole_in_g2_med": round(float(np.median([r["hole_in_g2"] for r in rr_])), 3)}
    rep["wall_s"] = round(time.time() - t0, 1)
    json.dump({"summary": rep, "gt": rows}, open(f"{OUT}/{rep['site']}.json", "w"), ensure_ascii=False, indent=0)
    return rep

S = []
sites = sorted(p[:-4] for p in glob.glob("/kaggle/input/**/*.las", recursive=True))
print("sites", len(sites), flush=True)
for site in sites:
    las = site + ".las"
    try:
        r = audit(site, las)
    except Exception as e:
        r = {"site": os.path.basename(site), "err": repr(e)[:500]}
    S.append(r); print(json.dumps({k: r.get(k) for k in ("site", "frac_ground", "pipeline_uses", "dz_all_minus_g2", "gt_povr", "gt_tselye", "err", "wall_s")}, ensure_ascii=False), flush=True)
    json.dump(S, open(f"{OUT}/summary.json", "w"), ensure_ascii=False, indent=1)
print("DONE", len(S), flush=True)
