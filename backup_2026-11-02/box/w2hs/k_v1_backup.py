# koz3-w2-hillshade-parity-s3 (CI2-8, W2). Свой код; вызывает только expds.lidar из репо.
# LAS (S3) -> DEM (expds.lidar.build_dem, шаг = шаг Li) -> hillshade_set -> compare_with_li против Li_{c,ch,g,i} организаторов.
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
        if not lis: summary.append({"las": k, "status": "нет Li"}); continue
        if time.time() + 1800 > DEADLINE: summary.append({"las": k, "status": "deadline"}); continue
        if (OUT / f"{tag}.json").exists(): continue
        shutil.rmtree(TMP, ignore_errors=True); (TMP / "li").mkdir(parents=True)
        t0 = time.time()
        try:
            lp = TMP / "a.las"; s3.download_file("train-expds-3", k, str(lp))
            for lk, _ in lis: s3.download_file("train-expds-3", lk, str(TMP / "li" / Path(lk).name))
            g = sorted((TMP / "li").glob("*_g.tif")) or sorted((TMP / "li").glob("*.tif"))
            with rasterio.open(g[0]) as ds: res = float(ds.res[0])
            r = subprocess.run([sys.executable, "-m", "expds.lidar", "check", "--las", str(lp), "--li-dir", str(TMP / "li"),
                                "--res", str(res)], capture_output=True, text=True, timeout=3600, cwd=str(CODE))
            rep = {"las": k, "bytes": sz, "li_res": res, "rc": r.returncode, "wall_s": round(time.time() - t0, 1),
                   "stderr_tail": r.stderr[-1500:]}
            try: rep.update(json.loads(r.stdout[r.stdout.index("{"):]))
            except Exception: rep["stdout_tail"] = r.stdout[-1500:]
        except Exception as e:
            rep = {"las": k, "status": "error", "err": repr(e)[:500], "wall_s": round(time.time() - t0, 1)}
        (OUT / f"{tag}.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
        summary.append({kk: rep.get(kk) for kk in ("las", "rc", "status", "wall_s", "li_res")})
        print(json.dumps(summary[-1], ensure_ascii=False), flush=True)
        (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    shutil.rmtree(TMP, ignore_errors=True)
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print("DONE", len(summary), flush=True)

main()
