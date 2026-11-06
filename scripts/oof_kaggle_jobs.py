#!/usr/bin/env python3
"""План заданий OOF-инференса для Kaggle-ядра (GPU-5).

Потребители: scripts/scoreboard.py (--honest 10), density_thresholds.py,
oof_transform calib, bot4/bot7 калибровка порогов по слою.

Режим по умолчанию — ``weights``: сданные веса из configs/default.toml [models]
на всех участках трейна (не fold-OOF). Честность honest-10 тогда в подборе
порога на чужих семьях, не в весах. Fold-OOF (``nn_oof --predict-only/--resume``)
включается env OOF_MODE=fold, когда на input/S3 есть work/fold* с весами.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Iterable

# Имена выходов — как ждёт scoreboard / notes (см. scripts/scoreboard.py).
JOBS: list[dict] = [
    # out_name, weights (rel to models/), markup for nn_oof, optional predict_markups
    {"id": "spor_gor", "out": "spor_gor_oof.geojson", "weights": ["spor_gor_r18.pt"], "markup": "sp_or"},
    {"id": "spor_focal", "out": "spor_focal_oof.geojson", "weights": ["spor_focal_r18.pt"], "markup": "sp_or"},
    {"id": "spor_arch", "out": "spor_arch_oof.geojson", "weights": ["surf_arch_r18.pt"], "markup": "sp_or"},
    {"id": "spor_focal_wide", "out": "spor_focal_wide_oof.geojson", "weights": ["spor_focal_wide_r18.pt"], "markup": "sp_or"},
    {"id": "li_gor_mask", "out": "li_gor_mask_oof.geojson", "weights": ["li_gor_mask_r18.pt"], "markup": "li"},
    {"id": "li_fort_mask", "out": "li_fort_mask_oof.geojson", "weights": ["li_fort_mask_r18.pt"], "markup": "li"},
    {"id": "li_dev", "out": "li_dev_oof.geojson", "weights": ["li_dev_r18.pt"], "markup": "li"},
    {"id": "li_hard2", "out": "li_hard2_oof.geojson", "weights": ["li_r18_hard2.pt"], "markup": "li"},
    {"id": "ae_fort_mask", "out": "ae_fort_mask_oof.geojson", "weights": ["ae_fort_mask_r18.pt"], "markup": "ae"},
    {"id": "or_arch", "out": "or_arch_oof.geojson", "weights": ["surf_arch_r18.pt"], "markup": "or"},
]


def job_done_marker(work: Path, job_id: str) -> Path:
    return work / "jobs" / job_id / "done.json"


def job_out_path(work: Path, out_name: str) -> Path:
    return work / "oof" / out_name


def is_job_complete(work: Path, job: dict) -> bool:
    """Идемпотентность: done.json + выходной geojson с тем же sha256."""
    marker = job_done_marker(work, job["id"])
    out = job_out_path(work, job["out"])
    if not marker.is_file() or not out.is_file():
        return False
    try:
        meta = json.loads(marker.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return bool(meta.get("sha256")) and meta["sha256"] == sha256_file(out)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_done(work: Path, job: dict, sha: str, n_features: int, seconds: float) -> None:
    d = job_done_marker(work, job["id"])
    d.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "id": job["id"],
        "out": job["out"],
        "sha256": sha,
        "n_features": n_features,
        "seconds": round(seconds, 3),
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "weights": job["weights"],
        "markup": job["markup"],
    }
    d.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_status(out_root: Path, state: str, **extra) -> None:
    """STATUS.txt: первая строка RUNNING|DONE|FAILED (+ время), далее key=value."""
    out_root.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    lines = [state, f"ts={ts}"]
    for k, v in extra.items():
        lines.append(f"{k}={v}")
    (out_root / "STATUS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    payload = {"status": state, "ts": ts, **extra}
    (out_root / "STATUS.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def select_jobs(ids: Iterable[str] | None = None) -> list[dict]:
    if not ids:
        return list(JOBS)
    want = set(ids)
    found = [j for j in JOBS if j["id"] in want]
    missing = want - {j["id"] for j in found}
    if missing:
        raise KeyError(f"неизвестные job id: {sorted(missing)}")
    return found


def count_features(geojson_path: Path) -> int:
    data = json.loads(geojson_path.read_text(encoding="utf-8"))
    return len(data.get("features") or [])


def write_sha256sums(work: Path, jobs: list[dict]) -> Path:
    lines = []
    for job in jobs:
        out = job_out_path(work, job["out"])
        if out.is_file():
            lines.append(f"{sha256_file(out)}  oof/{job['out']}")
    path = work / "SHA256SUMS"
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return path


def synthetic_fc(n: int = 2) -> dict:
    """Крошечный GeoJSON для CPU smoke (без torch/S3)."""
    feats = []
    for i in range(n):
        x0, y0 = 4_200_000.0 + i * 100, 7_500_000.0
        ring = [[x0, y0], [x0 + 40, y0], [x0 + 40, y0 + 40], [x0, y0 + 40], [x0, y0]]
        feats.append({
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [ring]},
            "properties": {
                "fid": i,
                "class_name": "kurgany_tselye",
                "confidence": 0.9,
                "region_name": "smoke",
                "markup_type": "sp_or",
                "crs": "EPSG:3857",
            },
        })
    return {"type": "FeatureCollection", "features": feats}


def run_smoke(work: Path) -> dict:
    """CPU smoke: один synthetic job, STATUS, sha256, skip-on-rerun."""
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    write_status(work, "RUNNING", phase="smoke")
    job = select_jobs(["spor_focal"])[0]
    out = job_out_path(work, job["out"])
    out.parent.mkdir(parents=True, exist_ok=True)
    if is_job_complete(work, job):
        write_status(work, "DONE", phase="smoke", note="already_complete")
        return {"skipped": True, "sha256": sha256_file(out)}
    t0 = time.perf_counter()
    out.write_text(json.dumps(synthetic_fc(2), ensure_ascii=False), encoding="utf-8")
    sha = sha256_file(out)
    n = count_features(out)
    write_done(work, job, sha, n, time.perf_counter() - t0)
    write_sha256sums(work, [job])
    # second call must skip
    assert is_job_complete(work, job)
    write_status(work, "DONE", phase="smoke", n_features=n, sha256=sha)
    return {"skipped": False, "sha256": sha, "n_features": n}


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        print(json.dumps(run_smoke(Path(td)), ensure_ascii=False))
