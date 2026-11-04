# koz3-w2-hillshade-parity-s3 (CI2-8, W2). Свой код; вызывает только expds.lidar из репо.
# LAS (S3) -> DEM (expds.lidar.build_dem, шаг = шаг Li) -> hillshade_set -> compare_with_li против Li_{g,i} организаторов.
# v2 (30.10): все LAS бакета; полная гистограмма классов на каждый LAS (и без Li); подгонка стиля ci28_hs_fit (az/alt/z, Horn, σ) на каждом участке с Li.
import os, sys, json, time, glob, re, subprocess, shutil
from pathlib import Path
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "laspy[lazrs]", "rasterio", "boto3"], check=False)
CODE = Path(sorted(glob.glob("/kaggle/input/**/expds_core.abi3.so", recursive=True))[0]).parent
sys.path[:0] = [str(CODE)]
os.environ["PYTHONPATH"] = str(CODE)
OUT = Path("/kaggle/working/hsparity"); OUT.mkdir(parents=True, exist_ok=True)
TMP = Path("/tmp/hs"); DEADLINE = time.time() + 10.5 * 3600
def _s3_cred():
    import glob as _g
    kid = sec = None
    try:
        from kaggle_secrets import UserSecretsClient
        _u = UserSecretsClient()
        kid, sec = _u.get_secret("KOZ3_S3_ACCESS_KEY_ID"), _u.get_secret("KOZ3_S3_SECRET_ACCESS_KEY")
    except Exception:
        pass
    if not kid or not sec:
        for p in sorted(_g.glob("/kaggle/input/**/cred.json", recursive=True)):
            try:
                c = json.loads(Path(p).read_text(encoding="utf-8"))
            except Exception:
                continue
            kid = c.get("id") or c.get("aws_access_key_id"); sec = c.get("key") or c.get("aws_secret_access_key")
            if kid and sec:
                break
    if not kid or not sec:
        raise SystemExit("no S3 creds (koz3-s3-cred)")
    return kid, sec


import math, struct
import numpy as np
def class_hist(p):
    with open(p, "rb") as h: hd = h.read(375)
    off = struct.unpack("<I", hd[96:100])[0]; fmt = hd[104] & 0x3f; rl = struct.unpack("<H", hd[105:107])[0]
    n = struct.unpack("<I", hd[107:111])[0]
    if hd[25] >= 4 and n == 0: n = struct.unpack("<Q", hd[247:255])[0]
    m = np.memmap(p, dtype=np.uint8, mode="r", offset=off, shape=(n, rl)); cnt = np.zeros(256, np.int64)
    for a in range(0, n, 20_000_000):
        c = np.asarray(m[a:a + 20_000_000, 15 if fmt < 6 else 16]); cnt += np.bincount((c & 0x1f) if fmt < 6 else c, minlength=256)
    return {"fmt": fmt, "n": int(n), "classes": {str(k): int(v) for k, v in enumerate(cnt) if v}, "frac_ground": round(float(cnt[2] / max(n, 1)), 4)}

def hs_fit(las, gp, ip, maxpix=60e6):
    # == w2/tools/ci28_hs_fit.py (bot-W2 29.10), пути параметром
    import rasterio
    from rasterio.windows import from_bounds
    from rasterio.enums import Resampling
    from scipy import ndimage as ndi
    from expds import lidar
    src = rasterio.open(gp); lres = src.res[0]
    t = time.time(); d = lidar.build_dem(las, res=lres, max_pixels=int(maxpix)); tb = time.time() - t
    H, W = d.dem.shape; b = (d.left, d.top - H * d.res, d.left + W * d.res, d.top)
    win = from_bounds(*b, transform=src.transform).round_offsets().round_lengths()
    rd = lambda ds: ds.read(1, window=win, boundless=True, fill_value=255, out_shape=(H, W),
                            resampling=Resampling.average if d.res > lres + 1e-9 else Resampling.nearest)
    G = rd(src); I = rd(rasterio.open(ip)) if ip else 255 - G
    valid = np.isfinite(d.dem); z = np.where(valid, d.dem, np.nanmin(d.dem)).astype(np.float32)
    ok = valid & (G < 255) & ndi.binary_erosion(valid, iterations=3)
    rng = np.random.default_rng(0); sel = np.flatnonzero(ok.ravel())
    if sel.size < 1000: return {"li_res": lres, "dem_res": d.res, "status": "too few valid px", "n_valid": int(sel.size)}
    sel = rng.choice(sel, min(100000, len(sel)), replace=False)
    def grads(zz, res, horn):
        if horn:
            k = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], np.float32) / (8 * res)
            gx = -ndi.correlate(zz, k, mode="nearest"); gy = ndi.correlate(zz, k.T, mode="nearest")
        else:
            gx = np.gradient(zz, res, axis=1); gy = -np.gradient(zz, res, axis=0)
        return gx.ravel()[sel], gy.ravel()[sel]
    def shade(gx, gy, az, alt, zf):
        sl = np.arctan(np.hypot(gx, gy) * zf); asp = np.arctan2(-gx, -gy); zen = math.radians(90 - alt)
        return math.cos(zen) * np.cos(sl) + math.sin(zen) * np.sin(sl) * np.cos(math.radians(az) - asp)
    def fit(target, gx, gy, grid):
        y = target.ravel()[sel].astype(np.float64); best = None
        for az in grid["az"]:
            for alt in grid["alt"]:
                for zf in grid["z"]:
                    s_ = shade(gx, gy, az, alt, zf); A = np.c_[s_, np.ones_like(s_)]; coef, *_ = np.linalg.lstsq(A, y, rcond=None)
                    p_ = np.clip(A @ coef, 0, 255); mae = float(np.abs(p_ - y).mean()); r = float(np.corrcoef(s_, y)[0, 1])
                    if best is None or mae < best["mae"]:
                        best = dict(az=az, alt=alt, z=zf, a=round(coef[0], 2), b=round(coef[1], 2), mae=round(mae, 3), r=round(r, 4))
        return best
    out = {"li_res": lres, "dem_res": d.res, "shape": [H, W], "build_s": round(tb, 1), "used_all": bool(d.info.get("used_all"))}
    coarse = dict(az=list(range(0, 360, 15)), alt=[15, 25, 35, 45, 55, 65, 75], z=[0.5, 1, 2, 3, 5])
    for horn in (False, True):
        gx, gy = grads(z, d.res, horn)
        for nm, T in (("g", G), ("i", I)):
            bc = fit(T, gx, gy, coarse)
            fine = dict(az=[bc["az"] + k for k in (-10, -5, 0, 5, 10)], alt=[max(5, bc["alt"] + k) for k in (-8, -4, 0, 4, 8)],
                        z=[bc["z"] * f for f in (0.7, 0.85, 1, 1.2, 1.4)])
            out[f'{nm}_{"horn" if horn else "grad"}'] = fit(T, gx, gy, fine)
        gx0, gy0 = gx, gy
    s_ = np.clip(np.rint(255 * shade(gx0, gy0, 315, 45, 1)), 0, 255)
    out["ours_default_g_mae"] = round(float(np.abs(s_ - G.ravel()[sel]).mean()), 3)
    out["ours_default_i_mae_255minus"] = round(float(np.abs((255 - s_) - I.ravel()[sel]).mean()), 3)
    bg = out["g_horn"]
    for sg in (1, 2, 3):
        gx, gy = grads(ndi.gaussian_filter(z, sg), d.res, True); s_ = shade(gx, gy, bg["az"], bg["alt"], bg["z"]); y = G.ravel()[sel]
        A = np.c_[s_, np.ones_like(s_)]; c, *_ = np.linalg.lstsq(A, y, rcond=None)
        out[f"g_horn_smooth{sg}px"] = dict(mae=round(float(np.abs(np.clip(A @ c, 0, 255) - y).mean()), 3), a=round(float(c[0]), 1), b=round(float(c[1]), 1))
    gi = G.ravel()[sel].astype(int) + I.ravel()[sel]; out["g_plus_i_med"] = float(np.median(gi))
    out["li_nodata255_frac"] = round(float((G == 255).mean()), 3); out["our_valid_frac"] = round(float(valid.mean()), 3)
    return out


def main():
    import boto3, rasterio
    from botocore.config import Config
    import expds_core; print("rust", expds_core.version(), flush=True)
    kid, sec = _s3_cred()
    s3 = boto3.client("s3", endpoint_url="https://s3.ru-7.storage.selcloud.ru", region_name="ru-7",
                      aws_access_key_id=kid, aws_secret_access_key=sec, config=Config(retries={"max_attempts": 8}))
    keys = []
    for pg in s3.get_paginator("list_objects_v2").paginate(Bucket="train-expds-3"):
        keys += [(o["Key"], o["Size"]) for o in pg.get("Contents", [])]
    las = sorted([(k, s) for k, s in keys if k.lower().endswith(".las")], key=lambda x: x[1])
    li = [(k, s) for k, s in keys if "_Li_карты/" in k and k.lower().endswith(".tif")]
    summary = []
    for k, sz in las:
        reg = k.split("/")[0]
        lis = [x for x in li if x[0].split("/")[0] == reg]
        tag = re.sub(r"[^\w-]+", "_", Path(k).stem)[:80]
        if time.time() + 1800 > DEADLINE: summary.append({"las": k, "status": "deadline"}); continue
        if (OUT / f"{tag}.json").exists(): continue
        shutil.rmtree(TMP, ignore_errors=True); (TMP / "li").mkdir(parents=True)
        t0 = time.time(); rep = {"las": k, "bytes": sz, "n_li": len(lis)}
        try:
            lp = TMP / "a.las"; s3.download_file("train-expds-3", k, str(lp))
            try: rep["cls"] = class_hist(str(lp))
            except Exception as e: rep["cls_err"] = repr(e)[:300]
            lis = [x for x in lis if re.search(r"_Li_(g|i)\.tif$", x[0])]
            if not lis:
                rep["status"] = "нет Li"
            else:
                for lk, _ in lis: s3.download_file("train-expds-3", lk, str(TMP / "li" / Path(lk).name))
                g = sorted((TMP / "li").glob("*_g.tif")) or sorted((TMP / "li").glob("*.tif"))
                ii = sorted((TMP / "li").glob("*_i.tif"))
                with rasterio.open(g[0]) as ds: res = float(ds.res[0])
                rep["li_res"] = res
                r = subprocess.run([sys.executable, "-m", "expds.lidar", "check", "--las", str(lp), "--li-dir", str(TMP / "li"),
                                    "--res", str(res)], capture_output=True, text=True, timeout=3600, cwd=str(CODE))
                rep.update({"rc": r.returncode, "stderr_tail": r.stderr[-1500:]})
                try: rep["check"] = json.loads(r.stdout[r.stdout.index("{"):])
                except Exception: rep["stdout_tail"] = r.stdout[-1500:]
                try: rep["fit"] = hs_fit(str(lp), str(g[0]), str(ii[0]) if ii else None)
                except Exception as e: rep["fit_err"] = repr(e)[:500]
            rep["wall_s"] = round(time.time() - t0, 1)
        except Exception as e:
            rep.update({"status": "error", "err": repr(e)[:500], "wall_s": round(time.time() - t0, 1)})
        (OUT / f"{tag}.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
        summary.append({**{kk: rep.get(kk) for kk in ("las", "rc", "status", "wall_s", "li_res")}, "frac_ground": (rep.get("cls") or {}).get("frac_ground"), "fit_g": (rep.get("fit") or {}).get("g_horn")})
        print(json.dumps(summary[-1], ensure_ascii=False), flush=True)
        (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    shutil.rmtree(TMP, ignore_errors=True)
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print("DONE", len(summary), flush=True)

main()
