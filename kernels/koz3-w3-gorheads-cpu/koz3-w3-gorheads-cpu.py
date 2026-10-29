"""W3 / KOZ-3: retrain gor heads (gor_v3_head.npz + gor_dense_head.npz) on Kaggle CPU.
Recipe = scripts/gor_heads/train_heads.py @ 19ccc9cf (bot-5), resumable; DINOv2-S from HF (internet on).
Reference sha256 (PC run 9f62a96, WEIGHTS_MANIFEST_v4 40fa5997):
  gor_v3_head.npz    5bd88f69b3ffbfdc5312214b4d6012c92be12c66539b7bf8ad35564c76a75e7d
  gor_dense_head.npz feb1591bb1e9ada63b32829178eae44a7e19fdfd7e99d7d222e3b1389c5b2cb7
"""
import os, sys, json, glob, hashlib, subprocess, tarfile, time, importlib.util
from pathlib import Path
W = Path("/kaggle/working"); t0 = time.time()
def log(*a): print(f"[{time.time()-t0:7.0f}s]", *a, flush=True)
# collect input datasets (support /kaggle/input/<slug> and nested layouts)
inp = W / "inp"; inp.mkdir(exist_ok=True)
for p in glob.glob("/kaggle/input/**/", recursive=True):
    p = Path(p)
    if len(p.parts) > 7: continue
    if p.name.startswith("koz3-") and not (inp / p.name).exists():
        os.symlink(p, inp / p.name)
log("inputs:", sorted(x.name for x in inp.iterdir()))
code = inp / "koz3-w3-gorheads-code"
repo = W / "repo"; repo.mkdir(exist_ok=True)
with tarfile.open(code / "gorheads_code.tar.gz") as tf: tf.extractall(repo, filter="data")
try:
    import rasterio, pyproj, transformers  # noqa
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "rasterio", "pyproj"])
os.environ["PIXGBM_REBUILD_ALL"] = "1"
spec = importlib.util.spec_from_file_location("rtt", code / "rebuild_train_tree.py")
rtt = importlib.util.module_from_spec(spec); spec.loader.exec_module(rtt)
data = W / "train-expds-3"
rep = rtt.rebuild_train_tree(dest=data, input_root=inp)
(W / "rebuild_report.json").write_text(json.dumps(rep, indent=1, default=str))
spor = sorted(glob.glob(f"{data}/*/*SpOr*/*.tif*")); utm = glob.glob(f"{data}/*/UTM.json")
log(f"SpOr tifs={len(spor)} UTM.json={len(utm)}")
assert spor and utm, "no SpOr/UTM after rebuild"
env = {**os.environ, "DATA": str(data), "MODEL": "facebook/dinov2-small", "WORK": str(W / "gor_heads"),
       "DEVICE": "cpu", "THREADS": str(os.cpu_count() or 4), "SEED": "0", "PYTHONPATH": str(repo)}
rc = subprocess.call([sys.executable, str(repo / "scripts/gor_heads/train_heads.py")], env=env, cwd=repo)
log("train_heads rc", rc)
ref = {"gor_v3_head.npz": "5bd88f69b3ffbfdc5312214b4d6012c92be12c66539b7bf8ad35564c76a75e7d",
       "gor_dense_head.npz": "feb1591bb1e9ada63b32829178eae44a7e19fdfd7e99d7d222e3b1389c5b2cb7"}
out = {"rc": rc, "heads": {}}
for n, s in ref.items():
    f = W / "gor_heads" / "heads" / n
    if f.exists():
        h = hashlib.sha256(f.read_bytes()).hexdigest()
        out["heads"][n] = {"sha256": h, "bytes": f.stat().st_size, "matches_pc": h == s}
log(json.dumps(out, indent=1)); (W / "W3_GORHEADS_RESULT.json").write_text(json.dumps(out, indent=1))
# drop the symlinked tree from outputs
import shutil; shutil.rmtree(data, ignore_errors=True); shutil.rmtree(inp, ignore_errors=True)
sys.exit(0 if rc == 0 and len(out["heads"]) == 2 else 1)
