#!/usr/bin/env python3
"""OOF ансамбля лидара c5-in + c12-in прод-кодом (PREREG kaggle/2026-11-05/li_ens_in/PREREG.md).

    python ens_oof.py RELIEF12_DIR OUT_DIR --c5 DIR5 --c12 DIR12 [--sites S1,S2] [--only-c5]

RELIEF12_DIR — вход c12 по участкам (`<участок>.npy` u8 [12,H,W] + `.json` с left/top/step/crs; локальная копия
dmitrijkv/koz3-cnx-li-prep12). Участок берёт веса своего фолда (region_fold.json ядер): `weights_cnx_c5-in_f{f}.pt`
из DIR5 и `weights_cnx_c12-in_f{f}.pt` из DIR12. Карты — `expds.lidar_cnx_head.predict_maps` (логиты двух сетей и 8 видов
D8 усредняются), курганы — `decode_kurgans` + `to_features`, как в проде. Выход: `OUT_DIR/oof_cnx_e512-in_f{f}.geojson`
и `region_fold.json` (формат OOF ядер, для gate_cnxin_ctx.py с CNXIN_VARIANT=e512-in).
`--only-c5` — санити PREREG: одна модель c5 (без c12) должна совпасть с OOF ядра c5-in, как parity_head.json.
Участок уже посчитан (`OUT_DIR/parts/<участок>.json`) — пропускается: прогон можно продолжать после обрыва.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
KUR = ["kurgany_povrezhdennye", "kurgany_tselye"]


def region_folds(*dirs: Path) -> tuple[dict, dict]:
    rf, fam = {}, {}
    for d in dirs:
        for p in sorted(glob.glob(str(d / "**" / "region_fold.json"), recursive=True)):
            j = json.loads(Path(p).read_text(encoding="utf-8"))
            for r, f in j["region_fold"].items():
                if rf.setdefault(r, int(f)) != int(f):
                    raise SystemExit(f"region_fold расходится по {r}")
            fam.update(j.get("region_family", {}))
    return rf, fam


def weight(d: Path, variant: str, f: int) -> Path:
    w = sorted(glob.glob(str(d / "**" / f"weights_cnx_{variant}_f{f}.pt"), recursive=True))
    if len(w) != 1:
        raise SystemExit(f"веса {variant} f{f}: найдено {len(w)} в {d}")
    return Path(w[0])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("relief", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--c5", type=Path, required=True)
    ap.add_argument("--c12", type=Path, default=None)
    ap.add_argument("--sites", default="")
    ap.add_argument("--only-c5", action="store_true")
    ap.add_argument("--device", default="cpu", help="cpu | cuda (Kaggle T4: fp32, как санити на CPU)")
    a = ap.parse_args()
    from expds import lidar_cnx as LC
    from expds import lidar_cnx_head as LH

    if not a.only_c5 and a.c12 is None:
        raise SystemExit("нужен --c12 (или --only-c5 для санити)")
    tag = "c5-in" if a.only_c5 else "e512-in"
    rf, fam = region_folds(a.c5) if a.only_c5 else region_folds(a.c5, a.c12)
    sites = [s for s in a.sites.split(",") if s] or sorted(rf)
    parts = a.out / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    segs: dict = {}

    def seg(variant: str, d: Path, f: int):
        key = (variant, f)
        if key not in segs:
            segs.clear() if len(segs) > 4 else None
            segs[key] = LC.CnxSegmenter([weight(d, variant, f)], device=a.device, opts={"classes": KUR}, amp=False)
        return segs[key]

    t_all = time.time()
    for site in sorted(sites, key=lambda r: rf[r]):
        done = parts / f"{site}.json"
        if done.is_file():
            continue
        f = rf[site]
        t0 = time.time()
        u8 = np.load(a.relief / f"{site}.npy")
        geo = json.loads((a.relief / f"{site}.json").read_text(encoding="utf-8"))
        geo.setdefault("crs", geo.get("crs"))
        x = (u8.astype(np.float32) - 128.0) / 127.0
        del u8
        s5 = seg("c5-in", a.c5, f)
        p = LH.params(s5.contract)
        valid = x[0] > LC.PAD_VALUE + 1e-6
        if a.only_c5:
            maps = LH.predict_maps(s5, x[:5], valid, p)
        else:
            s12 = seg("c12-in", a.c12, f)
            if LH.params(s12.contract) != p:
                raise SystemExit("разные параметры инференса c5/c12")
            maps = LH.predict_maps(s5, x[:5], valid, p, extra=[(s12, x)])
        dets = LH.decode_kurgans(maps, geo, p)
        feats = LH.to_features(dets, geo, p, site, f"{site}.npy", classes=KUR)
        rows = [{"type": "Feature", "geometry": ft["geometry"],
                 "properties": {"class_name": ft["class_name"], "confidence": ft["confidence"], "region_name": site,
                                "crs": "EPSG:3857", "markup_type": "li", "threshold_group": "li_cnx",
                                "original_crs": ft["original_crs"]}} for ft in feats]
        done.write_text(json.dumps({"fold": f, "features": rows}, ensure_ascii=False), encoding="utf-8")
        print(f"{site} f{f}: {len(rows)} курганов, {time.time() - t0:.0f} с", flush=True)
    by_fold: dict = {}
    for site in sites:
        d = json.loads((parts / f"{site}.json").read_text(encoding="utf-8"))
        by_fold.setdefault(d["fold"], []).extend(d["features"])
    for f, rows in sorted(by_fold.items()):
        for i, r in enumerate(rows):
            r["properties"]["fid"] = i
        (a.out / f"oof_cnx_{tag}_f{f}.geojson").write_text(
            json.dumps({"type": "FeatureCollection", "features": rows}, ensure_ascii=False), encoding="utf-8")
    (a.out / "region_fold.json").write_text(json.dumps({"region_fold": {s: rf[s] for s in sites},
                                                        "region_family": {s: fam.get(s, s) for s in sites}},
                                                       ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"sites": len(sites), "folds": sorted(by_fold), "sec": round(time.time() - t_all)}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
