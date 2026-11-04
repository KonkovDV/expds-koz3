#!/usr/bin/env python3
"""koz3-w4-li-post-full — full-stack honest-10 for [li_post] thr_nested / thr_nms8 vs B′.

CPU, internet-off. Datasets: elenakonkova/koz3-w2-gate + elenakonkova/koz3-w2-oof.
Sidecars (vendor/w5_rerank/scripts/data) embedded as gzip+base64 (Kaggle script
kernels upload only this code_file).

Contract
  goal: does Li-only thr_nested(+nms8) survive in the W5 stack vs B′=0.315800?
  hypothesis: production li thr/dedupe shrink layer Δ; expect ALL Δ ≪ +0.12
  criterion: Δ≥+0.003 vs B′, p_gt0≥0.9, no layer (li/ae/sp_or) drop >0.01, damaged ok
  P means: paired bootstrap p_gt0 — NOT precision
  thr: floors from nested s0 LOFO (train folds only) — DEFAULT_CLASS_TAU_FLOOR
  flags: [li_post] OFF in repo default.toml; this kernel applies levers explicitly
  budget: CPU ~20–60 min; branch bots/bot4-li-post; no LB submit
"""
from __future__ import annotations

import base64, gzip, io, json, os, shutil, subprocess, sys, time, hashlib, glob, tarfile
from pathlib import Path

T0 = time.time()
OUT = Path(os.environ.get("LI_POST_FULL_OUT", "/kaggle/working/li_post_full"))
OUT.mkdir(parents=True, exist_ok=True)
BOOT = int(os.environ.get("LI_POST_FULL_BOOT", "2000"))
BPRIME_SCORE = 0.315800
A_SCORE = 0.314615
PASS_D = 0.003
PASS_P = 0.9
LAYER_DROP_MAX = 0.01
FLOORS = {"kurgany_povrezhdennye": 0.30, "kurgany_tselye": 0.30}

# --- embedded sidecar blob; filled by build_embed.py ---
BUNDLE_GZ_B64 = r"""
<<<BUILD_EMBED>>>
"""


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


def extract_bundle(dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    blob = BUNDLE_GZ_B64.strip()
    if blob.startswith("<<<") or not blob:
        raise RuntimeError("BUNDLE_GZ_B64 not embedded — run build_embed.py")
    raw = gzip.decompress(base64.b64decode(blob))
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as tf:
        try:
            tf.extractall(dest, filter="data")
        except (TypeError, ValueError):
            tf.extractall(dest)


def resolve_here() -> Path:
    script_dir = Path(__file__).resolve().parent
    local_marker = script_dir / "vendor" / "materials" / "compute_metrics.py"
    if local_marker.is_file():
        log("HERE local package", script_dir)
        return script_dir
    dest = Path(os.environ.get("LI_POST_FULL_BUNDLE", "/kaggle/working/_lipost_bundle"))
    marker = dest / "vendor" / "materials" / "compute_metrics.py"
    if not marker.is_file():
        log("extract embedded sidecars →", dest)
        extract_bundle(dest)
    else:
        log("HERE cached bundle", dest)
    need = [
        "vendor/materials/compute_metrics.py",
        "vendor/expds/li_post.py",
        "vendor/expds/metric.py",
        "vendor/notes/las_boxes_59.json",
        "w5_rerank/prep.py",
        "w5_rerank/gate_v6c.py",
        "scripts/stack_apply_li_post.py",
    ]
    miss = [p for p in need if not (dest / p).is_file()]
    if miss:
        raise FileNotFoundError(f"sidecar missing after extract: {miss}")
    return dest


def find_w5base() -> Path:
    roots = [Path(os.environ.get("KAGGLE_INPUT", "/kaggle/input")), Path("/kaggle/input")]
    cands: list[Path] = []
    env = os.environ.get("LI_POST_FULL_W5BASE")
    if env:
        cands.append(Path(env))
    for root in roots:
        if not root.is_dir():
            continue
        for pat in (
            "koz3-w2-gate/w5base",
            "datasets/elenakonkova/koz3-w2-gate/w5base",
            "**/w5base",
        ):
            cands += [Path(x) for x in glob.glob(str(root / pat), recursive=True)]
        for p in glob.glob(str(root / "**/stack_arch.geojson"), recursive=True):
            cands.append(Path(p).parent)
    seen = set()
    for p in cands:
        try:
            rp = p.resolve()
        except Exception:
            continue
        if rp in seen:
            continue
        seen.add(rp)
        if (p / "stack_arch.geojson").is_file() and (p / "gt_train_v2.geojson").is_file():
            return p
    raise FileNotFoundError(
        "w5base/stack_arch.geojson not found (need elenakonkova/koz3-w2-gate "
        "or LI_POST_FULL_W5BASE). Box often lacks stack_arch/base_counts/bprime_counts; "
        "fam0 may be counts-only without jsonl."
    )


def find_counts(w5: Path, here: Path):
    for p in (
        here / "data" / "bprime_counts.pkl",
        w5 / "bprime_counts.pkl",
        w5 / "work" / "bprime_counts.pkl",
        w5 / "Bprime" / "base_counts.pkl",
    ):
        if p.is_file():
            return p, "Bprime", BPRIME_SCORE
    for p in (w5 / "base_counts.pkl", w5 / "work" / "base_counts.pkl"):
        if p.is_file():
            return p, "A", A_SCORE
    raise FileNotFoundError(
        "base_counts.pkl / bprime_counts.pkl missing in w5base — cannot pair-gate vs B′/A"
    )


def setup_vendor_root(work: Path, here: Path) -> Path:
    root = work / "repo"
    if root.exists():
        shutil.rmtree(root)
    (root / "expds").mkdir(parents=True)
    (root / "materials").mkdir(parents=True)
    (root / "notes").mkdir(parents=True)
    for f in (here / "vendor" / "expds").glob("*.py"):
        shutil.copy(f, root / "expds" / f.name)
    src_metric = here / "vendor" / "materials" / "compute_metrics.py"
    shutil.copy(src_metric, root / "materials" / "compute_metrics.py")
    (root / "materials" / "__init__.py").write_text("")
    for f in (here / "vendor" / "notes").iterdir():
        if f.is_file():
            shutil.copy(f, root / "notes" / f.name)
    return root


def py_env(here: Path, root: Path, data: Path, stack_name: str) -> dict:
    env = os.environ.copy()
    env["W5_ROOT"] = str(root)
    env["W5_DATA"] = str(data)
    env["W5_STACK"] = stack_name
    # ijson + package vendor first
    env["PYTHONPATH"] = os.pathsep.join([
        str(here / "vendor"),
        str(here / "w5_rerank"),
        str(root),
        env.get("PYTHONPATH", ""),
    ])
    return env


def apply_li_post(here: Path, stack_src: Path, out: Path, class_tau: bool, nms_m: float) -> dict:
    cmd = [
        sys.executable, str(here / "scripts" / "stack_apply_li_post.py"),
        str(stack_src), str(out),
        "--floors", json.dumps(FLOORS),
    ]
    if class_tau:
        cmd.append("--class-tau")
    if nms_m > 0:
        cmd += ["--nms-m", str(nms_m)]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(here / "vendor") + os.pathsep + env.get("PYTHONPATH", "")
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    (OUT / f"apply_{out.stem}.log").write_text(r.stdout + "\n" + r.stderr)
    if r.returncode != 0:
        raise RuntimeError(f"apply failed: {r.stderr[-2000:]}\n{r.stdout[-1000:]}")
    try:
        return json.loads(r.stdout.strip().splitlines()[-1])
    except Exception:
        return {"raw": r.stdout[-500:]}


def run_prep(here: Path, root: Path, data: Path, stack_name: str, tag: str) -> None:
    env = py_env(here, root, data, stack_name)
    r = subprocess.run(
        [sys.executable, str(here / "w5_rerank" / "prep.py")],
        cwd=str(here / "w5_rerank"), env=env, capture_output=True, text=True,
    )
    (OUT / f"prep_{tag}.log").write_text(r.stdout + "\n" + r.stderr)
    if r.returncode != 0:
        raise RuntimeError(f"prep failed {tag}: {r.stderr[-2000:]}\n{r.stdout[-1000:]}")


def run_gate(here: Path, root: Path, data: Path, stack_name: str, vs: Path, tag: str) -> dict:
    env = py_env(here, root, data, stack_name)
    cmd = [
        sys.executable, str(here / "w5_rerank" / "gate_v6c.py"),
        "--vs", str(vs), "--boot", str(BOOT),
        "--save", f"counts_{tag}.pkl",
        "--out", f"w6c_gate_{tag}.json",
    ]
    log("gate", tag, "boot", BOOT)
    r = subprocess.run(cmd, cwd=str(here / "w5_rerank"), env=env, capture_output=True, text=True)
    (OUT / f"gate_{tag}.log").write_text(r.stdout + "\n" + r.stderr)
    if r.returncode != 0:
        raise RuntimeError(f"gate failed {tag}: {r.stderr[-2000:]}\n{r.stdout[-1000:]}")
    res_path = data / "work" / f"w6c_gate_{tag}.json"
    res = json.loads(res_path.read_text())
    shutil.copy(res_path, OUT / f"gate_{tag}.json")
    return res


def layer_scores(here: Path, root: Path, data: Path, stack_name: str) -> dict:
    """Honest-10 ALL restricted to markup prefixes li*/ae*/sp* (approx per-layer)."""
    env = py_env(here, root, data, stack_name)
    code = r'''
import os, json, sys
sys.path[:0] = os.environ["PYTHONPATH"].split(os.pathsep)
from common import *
import fast, numpy as np
gt = load(W / "gt_train_v2.geojson"); new, optical = sets(gt); del gt
T = fast.T
mk = T.get("mk")
grp = T["grp"]
N = len(T["conf"])
out = {}
for name, prefs in (
    ("li", ("li",)),
    ("ae", ("ae",)),
    ("sp_or", ("sp_",)),
):
    keep = np.zeros(N, dtype=bool)
    for i in range(N):
        g = str(grp[i] or "")
        m = str(mk[i] if mk is not None else "")
        if any(g.startswith(p) or m.startswith(p) for p in prefs):
            keep[i] = True
    if keep.sum() == 0:
        out[name] = {"score": None, "n": 0}
        continue
    s, counts, info, folds = honest(fast.curves(keep=keep), new, optical)
    out[name] = {"score": float(s[0]), "n": int(keep.sum()), "info": info}
print(json.dumps(out))
'''
    r = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(here / "w5_rerank"), env=env, capture_output=True, text=True,
    )
    (OUT / f"layers_{stack_name}.log").write_text(r.stdout + "\n" + r.stderr)
    if r.returncode != 0:
        log("WARN layer_scores failed", r.stderr[-500:])
        return {}
    try:
        return json.loads(r.stdout.strip().splitlines()[-1])
    except Exception as e:
        log("WARN layer parse", e)
        return {}


def main():
    (OUT / "STATUS.txt").write_text("RUNNING\n")
    log("li-post-full start boot", BOOT, "floors", FLOORS)
    log("B′ provenance: documented Ae-ON flags-off honest-10 constant 0.315800 "
        "(HANDOFF_NOW / tta-gate / sat2-gate); prefer bprime_counts.pkl else "
        "base_counts.pkl (A=0.314615) + score−B′ abs. Inputs: stack_arch + "
        "gt_train_v2 + family honest-10 via prep→gate_v6c; per-layer OOF live in "
        "koz3-w2-oof / stack; fam0 jsonl often MISSING on box (counts only).")
    here = resolve_here()
    log("HERE", here)
    try:
        w5 = find_w5base()
    except FileNotFoundError as e:
        log("MISSING_INPUT", e)
        summary = {
            "status": "MISSING_INPUT",
            "error": str(e),
            "Bprime": BPRIME_SCORE,
            "note": "Attach elenakonkova/koz3-w2-gate (w5base). Box: no stack_arch/"
                    "base_counts/bprime_counts; fam0 only _counts.json; w2-gate API 403 from box.",
        }
        (OUT / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
        with open(OUT / "STATUS.txt", "a", encoding="utf-8") as f:
            f.write("DONE MISSING_INPUT\n")
        return

    log("w5base", w5)
    work = OUT / "run"
    work.mkdir(parents=True, exist_ok=True)
    root = setup_vendor_root(work, here)
    data = work / "data"
    data.mkdir(exist_ok=True)
    (data / "work").mkdir(exist_ok=True)
    shutil.copy(w5 / "gt_train_v2.geojson", data / "gt_train_v2.geojson")
    stack_src = w5 / "stack_arch.geojson"
    vs, base_name, base_score_ref = find_counts(w5, here)
    log("vs", base_name, vs, "ref_score", base_score_ref)
    if base_name != "Bprime":
        log("WARN no bprime_counts.pkl — gating vs A; also report score−B′ abs", BPRIME_SCORE)

    variants = [
        ("baseline", False, 0.0),
        ("thr_nested", True, 0.0),
        ("thr_nms8", True, 8.0),
    ]
    results = []
    layer_base = None

    for tag, do_tau, nms in variants:
        stack_name = f"stack_{tag}.geojson"
        stack_out = data / stack_name
        if tag == "baseline":
            # identity copy (stream via apply with no levers)
            stats = apply_li_post(here, stack_src, stack_out, class_tau=False, nms_m=0.0)
        else:
            stats = apply_li_post(here, stack_src, stack_out, class_tau=do_tau, nms_m=nms)
        log("apply", tag, stats)
        run_prep(here, root, data, stack_name, tag)
        if os.environ.get("LI_POST_FULL_STOP_AFTER_PREP") == "1":
            log("STOP_AFTER_PREP", tag)
            continue
        res = run_gate(here, root, data, stack_name, vs, tag)
        res["tag"] = tag
        res["apply"] = stats
        res["base_name"] = base_name
        res["delta_vs_bprime_abs"] = float(res["score"]) - BPRIME_SCORE
        layers = layer_scores(here, root, data, stack_name)
        res["layers"] = layers
        if tag == "baseline":
            layer_base = layers
        results.append(res)
        log(
            tag, "score", res.get("score"), "d", res.get("d"),
            "dBprime_abs", res.get("delta_vs_bprime_abs"),
            "p", (res.get("boot") or {}).get("p_gt0"),
        )

    # layer drop vs baseline
    for res in results:
        drops = {}
        ok = True
        if layer_base and res.get("layers"):
            for k, v in res["layers"].items():
                b = (layer_base.get(k) or {}).get("score")
                c = (v or {}).get("score")
                if b is None or c is None:
                    drops[k] = None
                    continue
                dd = float(c) - float(b)
                drops[k] = dd
                if dd < -LAYER_DROP_MAX:
                    ok = False
        res["layer_delta_vs_baseline"] = drops
        res["layer_drop_ok"] = ok
        # pass vs B′: prefer paired if vs was Bprime; else use abs Δ + soft paired vs A
        d_b = res["delta_vs_bprime_abs"]
        p = float((res.get("boot") or {}).get("p_gt0") or 0)
        dam_ok = bool(res.get("damaged_ok", True))
        if base_name == "Bprime":
            res["pass_vs_bprime"] = bool(
                res.get("pass_full") and res.get("layer_drop_ok")
            )
        else:
            res["pass_vs_bprime"] = bool(
                d_b >= PASS_D and p >= PASS_P and dam_ok and res.get("layer_drop_ok")
            )

    summary = {
        "goal": "li_post thr_nested(+nms8) full-stack honest-10 vs B′",
        "protocol": "prep→gate_v6c honest-10; floors nested s0 LOFO train-only",
        "Bprime_how": (
            "B′=0.315800 = documented Ae-ON / flags-off honest-10 baseline "
            "(HANDOFF_NOW, tta-gate bprime variant, sat2-gate). Computed historically "
            "via stack(+Ae) → prep → gate_v6c family honest-10; artifact bprime_counts.pkl. "
            "A=0.314615 = base_counts.pkl on stack_arch without that Ae bump. "
            "This kernel: pair vs bprime_counts if present else vs A + score−0.315800 abs."
        ),
        "inputs": {
            "w5base": str(w5),
            "stack": "stack_arch.geojson",
            "gt": "gt_train_v2.geojson",
            "vs": str(vs),
            "base_name": base_name,
            "floors_nested_LOFO": FLOORS,
            "oof_dataset": "elenakonkova/koz3-w2-oof (attached; apply uses stack li* in-place)",
        },
        "risk": (
            "Li-only +0.12 may shrink: stack li* already passed production "
            "[thresholds_by_markup.*]; class_tau is extra floor; LB thr ≠ OOF base 0.15."
        ),
        "Bprime_score": BPRIME_SCORE,
        "boot": BOOT,
        "results": [],
        "runtime_s": time.time() - T0,
    }
    for r in results:
        b = r.get("boot") or {}
        summary["results"].append({
            "tag": r.get("tag"),
            "score": r.get("score"),
            "base": r.get("base"),
            "d": r.get("d"),
            "delta_vs_bprime_abs": r.get("delta_vs_bprime_abs"),
            "d_new34": r.get("d_new34"),
            "d_optical": r.get("d_optical"),
            "boot": {k: b.get(k) for k in ("p_gt0", "p5", "p50", "p95", "mean") if k in b},
            "per_class": r.get("per_class"),
            "layers": r.get("layers"),
            "layer_delta_vs_baseline": r.get("layer_delta_vs_baseline"),
            "layer_drop_ok": r.get("layer_drop_ok"),
            "pass_full": r.get("pass_full"),
            "pass_soft": r.get("pass_soft"),
            "pass_vs_bprime": r.get("pass_vs_bprime"),
            "apply": r.get("apply"),
        })

    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False, default=float))
    lines = []
    for p in sorted(OUT.glob("gate_*.json")) + [OUT / "summary.json"]:
        if p.is_file():
            lines.append(f"{sha256(p)}  {p.name}")
    (OUT / "SHA256SUMS").write_text("\n".join(lines) + "\n")
    log("DONE variants", [r.get("tag") for r in results])
    with open(OUT / "STATUS.txt", "a", encoding="utf-8") as f:
        f.write("DONE\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log("FAILED", type(e).__name__, e)
        with open(OUT / "STATUS.txt", "a", encoding="utf-8") as f:
            f.write(f"FAILED {type(e).__name__}: {e}\n")
        raise
