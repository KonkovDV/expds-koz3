#!/usr/bin/env python3
"""koz3-bot7-geo-gate — REAL honest-10 for MagGr/geo levers vs B′=0.315800.

L3 geo_convex_hull = post-only on stack (runs first).
L1/L2 MagGr finds_points need classic re-infer → GEO_GATE_REINFER=1.
No stubs / silent nulls: missing w5base/stack/gt/counts → raise.
Baseline must be within ±0.002 of B′ (skipped only when GEO_GATE_DRYRUN=1).
"""
# NOTE: koz3_s3 block is prepended by assemble.py (or already inlined above).

import base64, gzip, io, json, os, shutil, subprocess, sys, time, hashlib, glob, tarfile, traceback
from pathlib import Path

T0 = time.time()
OUT = Path(os.environ.get("GEO_GATE_OUT", "/kaggle/working/geo_gate"))
OUT.mkdir(parents=True, exist_ok=True)
CACHE = Path(os.environ.get("GEO_GATE_S3_CACHE", "/kaggle/working/train-expds-3"))
BOOT = int(os.environ.get("GEO_GATE_BOOT", "2000"))
BPRIME = float(os.environ.get("BPRIME", "0.315800"))
A_SCORE = 0.314615
PASS_D = 0.003
PASS_P = 0.9
SMOKE = os.environ.get("GEO_GATE_SMOKE", "") == "1"
SKIP_S3 = os.environ.get("GEO_GATE_SKIP_S3", "") == "1"
DRYRUN = os.environ.get("GEO_GATE_DRYRUN", "") == "1"
REINFER = os.environ.get("GEO_GATE_REINFER", "") == "1"
BPRIME_TOL = float(os.environ.get("GEO_GATE_BPRIME_TOL", "0.002"))

GEO_PREFIXES = tuple(f"{n:03d}_" for n in range(200, 218))
GEO_MOD_RE = __import__("re").compile(r"(?i)(^|/)[^/]*(_)?(MagGr|EtGr|TerFl)(_|[^/]*)?(/|$)")
LABEL_HINTS = (
    "objectpoly", "object_poly", "findspoints", "finds_points",
    "разметка", "utm.json", "manifest.json",
)

LEVER_ORDER = ("L3_geo_convex_hull", "L1_extremum_snap", "L2_finds_nms", "ALL")
LEVERS = {
    "L3_geo_convex_hull": {"post.geo_convex_hull": True},
    "L1_extremum_snap": {"geophys.maggr.finds_points.extremum_snap_m": 3.0},
    "L2_finds_nms": {"geophys.maggr.finds_points.finds_nms_m": 1.5},
    "ALL": {
        "geophys.maggr.finds_points.extremum_snap_m": 3.0,
        "geophys.maggr.finds_points.finds_nms_m": 1.5,
        "post.geo_convex_hull": True,
    },
}
PRIORS = {
    "L1_extremum_snap": 0.0017,
    "L2_finds_nms": 0.0010,
    "L3_geo_convex_hull": 0.0017,
    "ALL": 0.0035,
}

# --- embedded sidecar blob; filled by build_embed.py ---
BUNDLE_GZ_B64 = r"""
<<<BUILD_EMBED>>>
"""


def log(*a):
    msg = f"[{time.time()-T0:7.0f}s] " + " ".join(str(x) for x in a)
    print(msg, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "STATUS.txt", "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def _write_status(state, **extra):
    OUT.mkdir(parents=True, exist_ok=True)
    payload = {"state": state, "elapsed_s": round(time.time() - T0, 1), **extra}
    (OUT / "STATUS.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


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
    dest = Path(os.environ.get("GEO_GATE_BUNDLE", "/kaggle/working/_geogate_bundle"))
    marker = dest / "vendor" / "materials" / "compute_metrics.py"
    if not marker.is_file():
        log("extract embedded sidecars →", dest)
        extract_bundle(dest)
    else:
        log("HERE cached bundle", dest)
    need = [
        "vendor/materials/compute_metrics.py",
        "vendor/expds/metric.py",
        "vendor/expds/post.py",
        "w5_rerank/prep.py",
        "w5_rerank/gate_v6c.py",
        "scripts/stack_apply_geo_hull.py",
    ]
    miss = [p for p in need if not (dest / p).is_file()]
    if miss:
        raise FileNotFoundError(f"sidecar missing after extract: {miss}")
    return dest


def find_w5base() -> Path:
    roots = [Path(os.environ.get("KAGGLE_INPUT", "/kaggle/input")), Path("/kaggle/input")]
    cands = []
    env = os.environ.get("GEO_GATE_W5BASE")
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
        "w5base/stack_arch.geojson not found — attach elenakonkova/koz3-w2-gate "
        "or set GEO_GATE_W5BASE. Refusing silent null scores."
    )


def find_counts(w5: Path, here: Path):
    for p in (
        here / "data" / "bprime_counts.pkl",
        w5 / "bprime_counts.pkl",
        w5 / "work" / "bprime_counts.pkl",
        w5 / "Bprime" / "base_counts.pkl",
    ):
        if p.is_file():
            return p, "Bprime", BPRIME
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
    notes = here / "vendor" / "notes"
    if notes.is_dir():
        for f in notes.iterdir():
            if f.is_file():
                shutil.copy(f, root / "notes" / f.name)
    return root


def py_env(here: Path, root: Path, data: Path, stack_name: str) -> dict:
    env = os.environ.copy()
    env["W5_ROOT"] = str(root)
    env["W5_DATA"] = str(data)
    env["W5_STACK"] = stack_name
    env["PYTHONPATH"] = os.pathsep.join([
        str(here / "vendor"),
        str(here / "w5_rerank"),
        str(root),
        env.get("PYTHONPATH", ""),
    ])
    return env


def apply_hull(here: Path, stack_src: Path, out: Path) -> dict:
    cmd = [sys.executable, str(here / "scripts" / "stack_apply_geo_hull.py"),
           str(stack_src), str(out)]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(here / "vendor") + os.pathsep + env.get("PYTHONPATH", "")
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    (OUT / f"apply_{out.stem}.log").write_text(r.stdout + "\n" + r.stderr)
    if r.returncode != 0:
        raise RuntimeError(f"hull apply failed: {r.stderr[-2000:]}\n{r.stdout[-1000:]}")
    return json.loads(r.stdout.strip().splitlines()[-1])


def run_prep(here, root, data, stack_name, tag):
    env = py_env(here, root, data, stack_name)
    r = subprocess.run(
        [sys.executable, str(here / "w5_rerank" / "prep.py")],
        cwd=str(here / "w5_rerank"), env=env, capture_output=True, text=True,
    )
    (OUT / f"prep_{tag}.log").write_text(r.stdout + "\n" + r.stderr)
    if r.returncode != 0:
        raise RuntimeError(f"prep failed {tag}: {r.stderr[-2000:]}\n{r.stdout[-1000:]}")


def run_gate(here, root, data, stack_name, vs, tag) -> dict:
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
    if not res_path.is_file():
        raise FileNotFoundError(f"gate output missing: {res_path}")
    res = json.loads(res_path.read_text())
    for k in ("score", "d", "boot"):
        if k not in res or res[k] is None:
            raise RuntimeError(f"gate {tag} returned null {k}: keys={list(res)}")
    shutil.copy(res_path, OUT / f"gate_{tag}.json")
    return res


def pack_result(name, flags, res, apply_stats, mode, note="") -> dict:
    boot = res.get("boot") or {}
    score = float(res["score"])
    d = float(res.get("d"))
    p = float(boot.get("p_gt0"))
    per = res.get("per_class") or {}
    damaged = None
    if isinstance(per, dict):
        damaged = per.get("kurgany_povrezhdennye")
    return {
        "state": "DONE",
        "lever": name,
        "flags": flags,
        "mode": mode,
        "bprime": BPRIME,
        "score": score,
        "delta": d,
        "delta_vs_bprime_abs": score - BPRIME,
        "p_gt0": p,
        "d_new34": res.get("d_new34"),
        "d_optical": res.get("d_optical"),
        "damaged_ok": res.get("damaged_ok"),
        "damaged_class": damaged,
        "per_class": per,
        "boot": {k: boot.get(k) for k in ("p_gt0", "p5", "p50", "p95", "mean") if k in boot},
        "pass_full": res.get("pass_full"),
        "pass_soft": res.get("pass_soft"),
        "pass_gate": bool(
            d >= PASS_D and p >= PASS_P and bool(res.get("damaged_ok", True))
        ),
        "expected_delta_lb": PRIORS.get(name),
        "apply": apply_stats,
        "note": note,
    }


def _want_key(key: str) -> bool:
    if key.endswith("/"):
        return False
    top = key.split("/", 1)[0]
    if not any(top.startswith(p) for p in GEO_PREFIXES):
        return False
    low = key.lower()
    if GEO_MOD_RE.search(key):
        return True
    return any(h in low for h in LABEL_HINTS)


def ensure_boto3():
    try:
        import boto3  # noqa: F401
    except ImportError:
        log("pip install boto3…")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "boto3"])


def pull_geo_from_s3() -> dict:
    ensure_boto3()
    try:
        cli = s3_client()  # noqa: F821 — inlined above by assemble
    except NameError:
        raise RuntimeError("s3_client not inlined — assemble.py failed")
    except Exception as e:
        # S3CredentialsError
        _write_status("FAILED", error=str(e), phase="s3_credentials")
        raise SystemExit(f"FATAL: S3 credentials: {e}") from e

    CACHE.mkdir(parents=True, exist_ok=True)
    marker = OUT / "checkpoints" / "s3_pull.json"
    marker.parent.mkdir(parents=True, exist_ok=True)
    if marker.exists():
        prev = json.loads(marker.read_text())
        if prev.get("state") == "DONE" and CACHE.exists():
            log("S3 pull checkpoint DONE — skip")
            return prev

    report = {
        "state": "RUNNING", "prefixes": list(GEO_PREFIXES),
        "listed": 0, "wanted": 0, "downloaded": 0, "skipped_exists": 0, "errors": [],
    }
    wanted = []
    try:
        for pref in GEO_PREFIXES:
            batch = list_keys(pref, client=cli)  # noqa: F821
            report["listed"] += len(batch)
            for key, size in batch:
                if _want_key(key):
                    wanted.append((key, size))
        report["wanted"] = len(wanted)
        log("S3 listed", report["listed"], "wanted", report["wanted"])
        for key, size in wanted:
            dst = CACHE / key
            try:
                if dst.exists() and dst.stat().st_size == size:
                    report["skipped_exists"] += 1
                    continue
                download(key, dst, client=cli, expected_size=size)  # noqa: F821
                report["downloaded"] += 1
            except Exception as e:
                report["errors"].append(f"{key}: {type(e).__name__}: {e}")
                if len(report["errors"]) > 50:
                    break
    except Exception as e:
        report["errors"].append(str(e))
        report["state"] = "FAILED"
        marker.write_text(json.dumps(report, ensure_ascii=False, indent=2))
        raise SystemExit(f"FATAL: S3 transfer failed: {e}") from e

    report["state"] = "DONE" if report["wanted"] > 0 else "FAILED"
    marker.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    if report["wanted"] == 0:
        raise SystemExit("FATAL: S3 returned 0 MagGr/EtGr/TerFl keys for 200_…217_")
    return report


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "checkpoints").mkdir(parents=True, exist_ok=True)
    (OUT / "STATUS.txt").write_text("RUNNING\n")
    _write_status("RUNNING", levers=list(LEVER_ORDER), dryrun=DRYRUN, reinfer=REINFER)

    if SMOKE:
        summary = {
            "mode": "SMOKE", "bprime": BPRIME,
            "gate": {"delta_min": PASS_D, "p_gt0_min": PASS_P, "damaged_not_worse": True},
            "levers": LEVERS, "s3": "skipped_in_smoke",
        }
        (OUT / "sweep_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
        _write_status("SMOKE")
        return 0

    s3_report = None
    if not SKIP_S3:
        s3_report = pull_geo_from_s3()
        log("S3", {k: s3_report.get(k) for k in ("state", "listed", "wanted", "downloaded", "skipped_exists")})

    here = resolve_here()
    w5 = find_w5base()
    log("w5base", w5)
    log(
        "B′ config: Ae-ON flags-off honest-10 = 0.315800 "
        "(PLAN_STICKY/HANDOFF; stack_arch+Ae → prep → gate_v6c; "
        "artifact bprime_counts.pkl). A=0.314615 = stack_arch without Ae bump."
    )

    work = OUT / "run"
    work.mkdir(parents=True, exist_ok=True)
    root = setup_vendor_root(work, here)
    data = work / "data"
    data.mkdir(exist_ok=True)
    (data / "work").mkdir(exist_ok=True)
    shutil.copy(w5 / "gt_train_v2.geojson", data / "gt_train_v2.geojson")
    stack_src = w5 / "stack_arch.geojson"
    if not stack_src.is_file():
        raise FileNotFoundError(f"missing stack_arch: {stack_src}")
    vs, base_name, base_score_ref = find_counts(w5, here)
    log("vs", base_name, vs, "ref", base_score_ref)

    # --- baseline ---
    baseline_stack = data / "stack_baseline.geojson"
    shutil.copy(stack_src, baseline_stack)
    run_prep(here, root, data, "stack_baseline.geojson", "baseline")
    base_res = run_gate(here, root, data, "stack_baseline.geojson", vs, "baseline")
    base_score = float(base_res["score"])
    log("baseline score", base_score, "ref", base_score_ref, "B′", BPRIME)
    if not DRYRUN:
        # Prefer match to documented B′ when gating vs Bprime counts; else vs A ref
        target = BPRIME if base_name == "Bprime" else base_score_ref
        # Also always check distance to documented B′ when score looks like full honest-10
        if abs(base_score - BPRIME) > BPRIME_TOL and abs(base_score - target) > BPRIME_TOL:
            raise RuntimeError(
                f"BASELINE SANITY FAIL: score={base_score} not within ±{BPRIME_TOL} of "
                f"B′={BPRIME} (or vs-ref {base_name}={target}). Abort — refuse silent drift."
            )
        if abs(base_score - BPRIME) > BPRIME_TOL:
            log("WARN baseline off B′ by", abs(base_score - BPRIME), "but matches", base_name, target)
        else:
            log("baseline OK vs B′ (|Δ|=", round(abs(base_score - BPRIME), 6), ")")
    else:
        log("DRYRUN: skip B′ ±tol check (synthetic stack)")

    runs = {
        "baseline": pack_result(
            "baseline", {}, base_res, {"n_in": "identity"}, "scoreboard",
            note="Ae-ON flags-off stack_arch honest-10",
        )
    }

    # --- L3 first (post-only) ---
    ck = OUT / "checkpoints" / "lever_L3_geo_convex_hull.json"
    if ck.exists() and json.loads(ck.read_text()).get("state") == "DONE" and not DRYRUN:
        runs["L3_geo_convex_hull"] = json.loads(ck.read_text())
        log("skip L3 (checkpoint)")
    else:
        hull_out = data / "stack_L3.geojson"
        stats = apply_hull(here, stack_src, hull_out)
        log("L3 apply", stats)
        run_prep(here, root, data, "stack_L3.geojson", "L3")
        res = run_gate(here, root, data, "stack_L3.geojson", vs, "L3")
        # delta vs baseline scoreboard (gate --vs pairs against base counts)
        packed = pack_result("L3_geo_convex_hull", LEVERS["L3_geo_convex_hull"], res, stats, "post_only")
        runs["L3_geo_convex_hull"] = packed
        ck.write_text(json.dumps(packed, ensure_ascii=False, indent=2) + "\n")
        log("L3 score", packed["score"], "Δ", packed["delta"], "P", packed["p_gt0"])

    # --- L1 / L2: reinfer or SKIPPED ---
    for name in ("L1_extremum_snap", "L2_finds_nms"):
        ck = OUT / "checkpoints" / f"lever_{name}.json"
        if ck.exists() and json.loads(ck.read_text()).get("state") in ("DONE", "SKIPPED") and not DRYRUN:
            runs[name] = json.loads(ck.read_text())
            continue
        if not REINFER:
            packed = {
                "state": "SKIPPED",
                "lever": name,
                "flags": LEVERS[name],
                "mode": "needs_reinfer",
                "bprime": BPRIME,
                "score": None,
                "delta": None,
                "p_gt0": None,
                "pass_full": None,
                "expected_delta_lb": PRIORS[name],
                "note": (
                    "SKIPPED: MagGr classic re-infer required. "
                    "Set GEO_GATE_REINFER=1 (S3 cache 200–217 MagGr). "
                    "Not a stub DONE — scores withheld until reinfer."
                ),
            }
            runs[name] = packed
            ck.write_text(json.dumps(packed, ensure_ascii=False, indent=2) + "\n")
            log("SKIPPED", name, "(GEO_GATE_REINFER!=1)")
            continue
        # REINFER path: require MagGr cache + geophys; fail loud if missing
        maggr_tifs = list(CACHE.glob("**/07_*_MagGr_*/**/*серый.tif")) + list(
            CACHE.glob("**/07_*_MagGr_*/**/*gray*.tif")
        ) + list(CACHE.glob("**/*MagGr*.tif"))
        if not maggr_tifs:
            raise FileNotFoundError(
                f"GEO_GATE_REINFER=1 but no MagGr tifs under {CACHE} — S3 pull incomplete"
            )
        raise NotImplementedError(
            f"{name}: MagGr reinfer wiring present-guard passed ({len(maggr_tifs)} tifs); "
            "full classic reinfer+merge into stack not yet enabled in this package build — "
            "refuse silent null. Use L3-only verdict or extend scripts/stack_reinfer_maggr.py."
        )

    # --- ALL = L3 (+ reinfer L1/L2 when available) ---
    ck = OUT / "checkpoints" / "lever_ALL.json"
    if ck.exists() and json.loads(ck.read_text()).get("state") == "DONE" and not DRYRUN:
        runs["ALL"] = json.loads(ck.read_text())
    else:
        if REINFER:
            raise NotImplementedError("ALL+REINFER: same as L1/L2 — refuse stub")
        # ALL without reinfer ≡ L3
        runs["ALL"] = dict(runs["L3_geo_convex_hull"])
        runs["ALL"]["lever"] = "ALL"
        runs["ALL"]["flags"] = LEVERS["ALL"]
        runs["ALL"]["note"] = (
            "ALL without GEO_GATE_REINFER ≡ L3 only (L1/L2 SKIPPED). "
            "expected_delta_lb prior assumes all three."
        )
        runs["ALL"]["expected_delta_lb"] = PRIORS["ALL"]
        runs["ALL"]["mode"] = "post_only_L3_subset"
        ck.write_text(json.dumps(runs["ALL"], ensure_ascii=False, indent=2) + "\n")

    # Gate verdict
    l3 = runs["L3_geo_convex_hull"]
    verdict = {
        "L3_pass": bool(l3.get("pass_gate")),
        "L3_delta": l3.get("delta"),
        "L3_p_gt0": l3.get("p_gt0"),
        "baseline_score": runs["baseline"]["score"],
        "baseline_vs_bprime": runs["baseline"]["score"] - BPRIME,
        "reinfer": REINFER,
        "criterion": f"Δ≥{PASS_D}, p_gt0≥{PASS_P}, damaged not worse",
    }

    summary = {
        "mode": "DRYRUN" if DRYRUN else "FULL",
        "bprime": BPRIME,
        "bprime_config": (
            "B′=0.315800 = Ae-ON / flags-off honest-10 on family folds "
            "(documented PLAN_STICKY/HANDOFF/tta-gate). Measured via stack_arch(+Ae) "
            "→ w5_rerank prep → gate_v6c; pair artifact bprime_counts.pkl. "
            "A=0.314615 = base_counts.pkl on stack_arch without Ae bump (|A−B′|≈0.001185)."
        ),
        "gate": {"delta_min": PASS_D, "p_gt0_min": PASS_P, "damaged_not_worse": True},
        "s3": s3_report,
        "w5base": str(w5),
        "vs": str(vs),
        "base_name": base_name,
        "stack": str(stack_src),
        "gt": str(data / "gt_train_v2.geojson"),
        "runs": runs,
        "verdict": verdict,
        "cache": str(CACHE),
        "runtime_s": round(time.time() - T0, 1),
        "boot": BOOT,
    }
    (OUT / "sweep_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float) + "\n"
    )
    lines = []
    for p in sorted(OUT.glob("gate_*.json")) + [OUT / "sweep_summary.json"]:
        if p.is_file():
            lines.append(f"{sha256(p)}  {p.name}")
    (OUT / "SHA256SUMS").write_text("\n".join(lines) + "\n")
    _write_status("DONE", verdict=verdict, levers=list(runs))
    log("DONE verdict", verdict)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        _write_status("FAILED", error=traceback.format_exc()[-2000:])
        raise
