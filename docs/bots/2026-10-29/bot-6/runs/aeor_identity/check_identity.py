"""bot-6: флаг [dfine_aeor] выкл. → result.geojson побайтно как у v3 (360c870). Синтетический вход conftest
(LAS участка + MagGr), pipeline.run из каждого корня, свой конфиг. Запуск: python check_identity.py OUT_DIR"""
import hashlib, json, subprocess, sys, os
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE / "base"
BR = Path("/workspace/koz3_shared/bot-6/repo")
PY = sys.executable
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)

mk = f"""
import sys, json; sys.path.insert(0, {str(BR / 'tests')!r}); sys.path.insert(0, {str(BR)!r})
from pathlib import Path
from conftest import write_las, write_mag_tif
r1 = Path({str(out / 'input')!r}) / 'region_0001'
write_las(r1 / '01_region_0001_La_точки' / '01_region_0001_La_точки.las')
(r1 / 'UTM.json').write_text(json.dumps({{'utm': '36N'}}), encoding='utf-8')
r2 = Path({str(out / 'input')!r}) / 'region_0002'
write_mag_tif(r2 / '07_region_0002_MagGr_карты' / '01_region_0002_MagGr.tif')
"""
if not (out / "input").exists():
    subprocess.run([PY, "-c", mk], check=True)

def run(root: Path, cfg_text: str, tag: str):
    cfg = out / f"{tag}.toml"; cfg.write_text(cfg_text, encoding="utf-8")
    res = out / f"{tag}.geojson"
    code = f"import sys; sys.path.insert(0, {str(root)!r}); from expds import pipeline; pipeline.run({str(out / 'input')!r}, {str(res)!r}, {str(cfg)!r})"
    env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", PYTHONDONTWRITEBYTECODE="1", MALLOC_ARENA_MAX="2", RAYON_NUM_THREADS="2", NUMEXPR_MAX_THREADS="2")
    r = subprocess.run(["taskset", "-c", "0,1", PY, "-c", code], cwd=str(root), capture_output=True, text=True, env=env, timeout=1200)
    (out / f"{tag}.log").write_text(r.stdout + r.stderr, encoding="utf-8")
    assert r.returncode == 0, (tag, r.stderr[-2000:])
    return hashlib.sha256(res.read_bytes()).hexdigest(), r.stdout + r.stderr

base_cfg = (BASE / "configs/default.toml").read_text(encoding="utf-8")
br_cfg = (BR / "configs/default.toml").read_text(encoding="utf-8")
unready = br_cfg.replace("[dfine_aeor]\nenabled = false\n", "[dfine_aeor]\nenabled = true\n")
assert unready != br_cfg
h0, _ = run(BASE, base_cfg, "v3_base")
h1, _ = run(BR, br_cfg, "branch_off")
h2, log2 = run(BR, unready, "branch_on_unready")
rep = {"v3_base_360c870": h0, "branch_flag_off": h1, "branch_flag_on_no_weights": h2,
       "off_identical": h0 == h1, "unready_identical": h0 == h2,
       "unready_marker_logged": "DFINE_AEOR_UNAVAILABLE" in log2,
       "n_features": len(json.loads((out / "v3_base.geojson").read_text())["features"])}
(out / "identity.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))
print(json.dumps(rep, ensure_ascii=False, indent=1))
