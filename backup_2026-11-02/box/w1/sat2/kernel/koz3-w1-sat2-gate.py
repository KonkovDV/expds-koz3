#!/usr/bin/env python3
"""KOZ3 W1/bot2: sat2 REPLACE honest-10 (P≥0.9 vs B′/A) + sat_thr_relax scale sweep.

Slug: elenakonkova/koz3-w1-sat2-gate  — bot1 pushes (do NOT push from bot2).
Dataset: elenakonkova/koz3-w2-gate (w5base: stack_arch + gt + base_counts).
OOF/code: private dataset elenakonkova/koz3-w1-sat2-src (data/, w5_rerank/, vendor/, scripts/) → copied to
/kaggle/working/src (v2 01.11 bot-W1-aeor: v1 ERROR 0 s — script kernel ships one file, vendor/ was missing).

Contract
  goal: full honest-10 for sat2 REPLACE; then thr_relax scale sweep on top (honest, no LB leak)
  hypothesis: REPLACE soft-PASS 0.320516 ΔB′+0.00472 P0.838 reaches P≥0.9 with more boots
              and/or best sat_thr_relax scale on keep-mask
  criterion: Δ≥+0.003, boot P(Δ>0)=p_gt0≥0.9, damaged (kurgany_povrezhdennye) not worse,
             d_new34≥−0.001, d_optical≥−0.001  (gate_v6c.accept + damaged)
  P means: paired bootstrap p_gt0 (probability score(cand)>score(base)) — NOT precision
  budget: CPU ~25–60 min (1× prep REPLACE + N scale gates on keep-mask, boot=4000)
  branch: bots/w1-sat2-gate (lease bot-W1-sat); user alias bots/bot2-*
  default-OFF: sat_thr_relax in default.toml; REPLACE is candidate under test
  expected ΔLB: +0.003…+0.006 if pass_full (sat lever); hold official LB 0.41238 — no submit

Outputs /kaggle/working/sat2_gate/:
  STATUS.txt, gate_replace_all.json, gate_scale_*.json, sweep_summary.json, SHA256SUMS
"""
from __future__ import annotations

import json, os, shutil, subprocess, sys, time, hashlib, glob
from pathlib import Path

def _ensure_deps():
    import importlib
    miss = []
    for m in ("ijson", "shapely", "numpy"):
        try:
            importlib.import_module(m)
        except ImportError:
            miss.append(m)
    if miss:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *miss])
_ensure_deps()

T0 = time.time()
def resolve_src() -> Path:
    """Tree with vendor/materials/compute_metrics.py: beside the script, else from the attached dataset
    (extracted dir or uploaded .tar/.zip) copied/extracted to /kaggle/working/src."""
    import tarfile, zipfile
    here = Path(__file__).resolve().parent
    if (here / "vendor" / "materials" / "compute_metrics.py").is_file():
        return here
    dst = Path("/kaggle/working/src")
    if (dst / "vendor" / "materials" / "compute_metrics.py").is_file():
        return dst
    hits = sorted(glob.glob("/kaggle/input/**/vendor/materials/compute_metrics.py", recursive=True))
    if hits:
        root = Path(hits[0]).parents[2]
        shutil.copytree(root, dst, dirs_exist_ok=True)
        return dst
    for a in sorted(glob.glob("/kaggle/input/**/*sat2-src*.tar*", recursive=True) + glob.glob("/kaggle/input/**/src*.tar*", recursive=True)
                    + glob.glob("/kaggle/input/**/*.zip", recursive=True)):
        dst.mkdir(parents=True, exist_ok=True)
        if a.endswith(".zip"):
            zipfile.ZipFile(a).extractall(dst)
        else:
            tarfile.open(a).extractall(dst)
        for c in [dst] + [x for x in dst.iterdir() if x.is_dir()]:
            if (c / "vendor" / "materials" / "compute_metrics.py").is_file():
                return c
    raise FileNotFoundError("sat2 src (vendor/materials/compute_metrics.py) not found beside script or under /kaggle/input")


HERE = resolve_src()
OUT = Path(os.environ.get("SAT2_GATE_OUT", "/kaggle/working/sat2_gate"))
OUT.mkdir(parents=True, exist_ok=True)
BOOT = int(os.environ.get("SAT2_BOOT", "4000"))
SCALES = [float(x) for x in os.environ.get(
    "SAT2_SCALES", "1.0,0.95,0.9,0.85,0.8,0.75,0.7,0.6,0.5"
).split(",") if x.strip()]
BPRIME_SCORE = 0.315800
A_SCORE = 0.314615
PASS_D = 0.003
PASS_P = 0.9


def log(*a):
    msg = f"[{time.time()-T0:7.0f}s] " + " ".join(str(x) for x in a)
    print(msg, flush=True)
    with open(OUT / "STATUS.txt", "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_w5base() -> Path:
    cands = []
    for pat in (
        "/kaggle/input/koz3-w2-gate/w5base",
        "/kaggle/input/datasets/elenakonkova/koz3-w2-gate/w5base",
        "/kaggle/input/**/w5base",
    ):
        cands += [Path(x) for x in glob.glob(pat, recursive=True)]
    for pat in ("/kaggle/input/**/stack_arch.geojson",):
        for p in glob.glob(pat, recursive=True):
            cands.append(Path(p).parent)
    seen = set()
    for p in cands:
        rp = p.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        if (p / "stack_arch.geojson").is_file() and (p / "gt_train_v2.geojson").is_file():
            return p
    raise FileNotFoundError("w5base/stack_arch.geojson not found under /kaggle/input")


def find_counts(w5: Path):
    """Prefer B′ counts if present; else A base_counts.pkl."""
    for p in (
        HERE / "data" / "bprime_counts.pkl",
        w5 / "bprime_counts.pkl",
        w5 / "work" / "bprime_counts.pkl",
        w5 / "Bprime" / "base_counts.pkl",
        w5.parent / "bprime" / "base_counts.pkl",
    ):
        if p.is_file():
            return p, "Bprime", BPRIME_SCORE
    for p in (w5 / "base_counts.pkl", w5 / "work" / "base_counts.pkl"):
        if p.is_file():
            return p, "A", A_SCORE
    raise FileNotFoundError("base_counts.pkl missing in w5base")


def setup_vendor_root(work: Path) -> Path:
    root = work / "repo"
    if root.exists():
        shutil.rmtree(root)
    (root / "expds").mkdir(parents=True)
    (root / "materials").mkdir(parents=True)
    (root / "notes").mkdir(parents=True)
    for f in (HERE / "vendor" / "expds").glob("*.py"):
        shutil.copy(f, root / "expds" / f.name)
    shutil.copy(HERE / "vendor" / "materials" / "compute_metrics.py", root / "materials" / "compute_metrics.py")
    (root / "materials" / "__init__.py").write_text("")
    for f in (HERE / "vendor" / "notes").iterdir():
        if f.is_file():
            shutil.copy(f, root / "notes" / f.name)
    return root


def run_gate(data_dir: Path, root: Path, vs: Path, tag: str, boot: int,
             scale=None, thr_json: Path | None = None) -> dict:
    env = os.environ.copy()
    env["W5_ROOT"] = str(root)
    env["W5_DATA"] = str(data_dir)
    env["W5_STACK"] = os.environ.get("W5_STACK", "stack_replace_all.geojson")
    env["PYTHONPATH"] = (
        str(HERE / "w5_rerank") + os.pathsep + str(root) + os.pathsep + env.get("PYTHONPATH", "")
    )
    cmd = [
        sys.executable, str(HERE / "w5_rerank" / "gate_v6c.py"),
        "--vs", str(vs), "--boot", str(boot),
        "--save", f"counts_{tag}.pkl",
        "--out", f"w6c_gate_{tag}.json",
    ]
    if scale is not None and thr_json is not None:
        cmd += ["--scale", str(scale), "--thr-json", str(thr_json)]
    log("gate", tag, "boot", boot, "scale", scale)
    r = subprocess.run(cmd, cwd=str(HERE / "w5_rerank"), env=env, capture_output=True, text=True)
    (OUT / f"gate_{tag}.log").write_text(r.stdout + "\n" + r.stderr)
    if r.returncode != 0:
        raise RuntimeError(f"gate failed {tag}: {r.stderr[-2000:]}\n{r.stdout[-1000:]}")
    res_path = data_dir / "work" / f"w6c_gate_{tag}.json"
    res = json.loads(res_path.read_text())
    shutil.copy(res_path, OUT / f"gate_{tag}.json")
    return res


def main():
    (OUT / "STATUS.txt").write_text("RUNNING\n")
    log("sat2-gate start boot", BOOT, "scales", SCALES)
    log("src", HERE)
    w5 = find_w5base()
    log("w5base", w5)
    work = OUT / "run"
    work.mkdir(parents=True, exist_ok=True)
    root = setup_vendor_root(work)
    data = work / "data"
    data.mkdir(exist_ok=True)
    (data / "work").mkdir(exist_ok=True)
    shutil.copy(w5 / "gt_train_v2.geojson", data / "gt_train_v2.geojson")
    stack_src = w5 / "stack_arch.geojson"
    vs, base_name, base_score_ref = find_counts(w5)
    log("vs", base_name, vs, "ref_score", base_score_ref)
    if base_name != "Bprime":
        log("WARN no bprime_counts.pkl — gating vs A; also report score−B′ absolute", BPRIME_SCORE)

    oof = HERE / "data" / "sp_det2_oof_notta.geojson"
    thr_json = HERE / "data" / "sp_or_default_thr.json"
    assert oof.is_file(), oof
    n_oof = len(json.load(open(oof, encoding="utf-8"))["features"])
    log("oof feats", n_oof, "sha", sha256(oof)[:16])

    # REPLACE (drop sp_* kurgans + append all OOF) — soft-PASS reproduction path
    stack_all = data / "stack_replace_all.geojson"
    cmd = [
        sys.executable, str(HERE / "scripts" / "stack_replace_sp.py"),
        str(stack_src), str(oof), str(stack_all),
    ]
    log("replace_all", "…")
    subprocess.check_call(cmd)

    env = os.environ.copy()
    env["W5_ROOT"] = str(root)
    env["W5_DATA"] = str(data)
    env["W5_STACK"] = "stack_replace_all.geojson"
    env["PYTHONPATH"] = (
        str(HERE / "w5_rerank") + os.pathsep + str(root) + os.pathsep + env.get("PYTHONPATH", "")
    )
    log("prep replace_all (once)")
    r = subprocess.run(
        [sys.executable, str(HERE / "w5_rerank" / "prep.py")],
        cwd=str(HERE / "w5_rerank"), env=env, capture_output=True, text=True,
    )
    (OUT / "prep_replace_all.log").write_text(r.stdout + "\n" + r.stderr)
    if r.returncode != 0:
        raise RuntimeError(f"prep failed: {r.stderr[-2000:]}")
    log("prep done")

    results = []
    # 1) unfiltered REPLACE
    res = run_gate(data, root, vs, "replace_all", BOOT)
    res["tag"] = "replace_all"
    res["scale"] = None
    res["base_name"] = base_name
    res["delta_vs_bprime_abs"] = float(res["score"]) - BPRIME_SCORE
    results.append(res)
    log(
        "replace_all score", res.get("score"), "d", res.get("d"),
        "p", (res.get("boot") or {}).get("p_gt0"),
        "pass_full", res.get("pass_full"), "pass_soft", res.get("pass_soft"),
    )

    # 2) sat_thr_relax scale sweep via keep-mask on same table (no re-prep; honest nested)
    for sc in SCALES:
        tag = f"scale_{str(sc).replace('.', 'p')}"
        res = run_gate(data, root, vs, tag, BOOT, scale=sc, thr_json=thr_json)
        res["tag"] = tag
        res["scale"] = sc
        res["base_name"] = base_name
        res["delta_vs_bprime_abs"] = float(res["score"]) - BPRIME_SCORE
        results.append(res)
        log(
            tag, "score", res.get("score"), "d", res.get("d"),
            "p", (res.get("boot") or {}).get("p_gt0"),
            "damaged_dF1", (res.get("per_class") or {}).get("kurgany_povrezhdennye"),
            "pass_full", res.get("pass_full"),
        )

    def key(r):
        return (
            bool(r.get("pass_full")),
            float((r.get("boot") or {}).get("p_gt0") or 0),
            float(r.get("d") or -9),
        )
    best = max(results, key=key)
    summary = {
        "goal": "sat2 REPLACE honest-10 P>=0.9 + sat_thr_relax sweep",
        "hypothesis": "soft-PASS +0.00472 P0.838 → full P>=0.9; scale may lift P/Δ",
        "criterion": {
            "d": PASS_D, "p_gt0": PASS_P, "damaged_not_worse": True,
            "d_new34_min": -0.001, "d_optical_min": -0.001,
        },
        "P_means": "paired bootstrap p_gt0 = P(score_cand > score_base); NOT precision",
        "base_name": base_name,
        "base_score_ref": base_score_ref,
        "Bprime_score": BPRIME_SCORE,
        "boot": BOOT,
        "soft_PASS_ref": {"score": 0.320516, "d_Bprime": 0.00472, "p": 0.838},
        "oof_sha256": sha256(oof),
        "oof_n": n_oof,
        "note_oof": "merged folds0-4 no-TTA sha d28feee1…; Drive soft-PASS zip sha c1e54f76… (may differ)",
        "results": [],
        "best": {
            "tag": best.get("tag"), "scale": best.get("scale"),
            "score": best.get("score"), "d": best.get("d"),
            "boot": {k: (best.get("boot") or {}).get(k) for k in ("p_gt0", "p5", "p50", "p95", "mean")},
            "pass_full": best.get("pass_full"), "pass_soft": best.get("pass_soft"),
            "delta_vs_bprime_abs": best.get("delta_vs_bprime_abs"),
            "per_class": best.get("per_class"),
        },
        "expected_delta_LB": (
            "+0.003..+0.006 on sat lever if pass_full; else 0 (hold 0.41238, no submit)"
        ),
        "runtime_s": time.time() - T0,
    }
    for r in results:
        b = r.get("boot") or {}
        summary["results"].append({
            "tag": r.get("tag"), "scale": r.get("scale"),
            "score": r.get("score"), "base": r.get("base"), "d": r.get("d"),
            "d_new34": r.get("d_new34"), "d_optical": r.get("d_optical"),
            "boot": {k: b.get(k) for k in ("p_gt0", "p5", "p50", "p95", "mean") if k in b},
            "per_class": r.get("per_class"),
            "pass_full": r.get("pass_full"), "pass_soft": r.get("pass_soft"),
            "accept": r.get("accept"),
            "delta_vs_bprime_abs": r.get("delta_vs_bprime_abs"),
        })
    (OUT / "sweep_summary.json").write_text(json.dumps(summary, indent=1, default=float))
    lines = []
    for p in sorted(OUT.glob("gate_*.json")) + [OUT / "sweep_summary.json"]:
        if p.is_file():
            lines.append(f"{sha256(p)}  {p.name}")
    (OUT / "SHA256SUMS").write_text("\n".join(lines) + "\n")
    log(
        "BEST", best.get("tag"), "score", best.get("score"), "d", best.get("d"),
        "p", (best.get("boot") or {}).get("p_gt0"), "pass_full", best.get("pass_full"),
    )
    log("DONE")
    with open(OUT / "STATUS.txt", "a", encoding="utf-8") as f:
        f.write(f"DONE best={best.get('tag')} pass_full={best.get('pass_full')}\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log("FAILED", type(e).__name__, e)
        with open(OUT / "STATUS.txt", "a", encoding="utf-8") as f:
            f.write(f"FAILED {type(e).__name__}: {e}\n")
        raise
