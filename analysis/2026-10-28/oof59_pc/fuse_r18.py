"""Копии кругов лидара в аэрофото тем же правилом, что F:\\expds3\\work\\fuse_li59.py (27.10).
Отличие только в путях: вход и выход — аргументы, пишем под C:\\Sloi\\oof59."""
import json, sys
from collections import defaultdict
from pathlib import Path

REPO = Path(r"C:\Sloi\expds-koz3_v0.2.0\expds-koz3")
sys.path.insert(0, str(REPO))
from shapely.geometry import shape
from expds import writer
from expds.catalog import read_utm_json, scan_root
from expds.fusion import FusionParams, li_to_ae
from expds.pipeline import _params, load_config

ROOT = Path(r"F:\2\train-expds-3")
CFG = REPO / "configs" / "default.toml"


def main(src: Path, dst: Path) -> None:
    fp = _params(FusionParams, load_config(CFG).get("fusion"))
    raw = json.loads(src.read_text(encoding="utf-8"))
    by = defaultdict(list)
    for feature in raw["features"]:
        props = feature.get("properties") or {}
        by[str(props.get("region_name"))].append({
            "geom": shape(feature["geometry"]),
            "class_name": props.get("class_name"),
            "confidence": float(props.get("confidence") or 0),
            "region_name": props.get("region_name"),
            "markup_type": props.get("markup_type") or "li",
            "original_crs": props.get("original_crs"),
            "source": props.get("source"),
        })
    del raw
    out, n_copy, n_ae = [], 0, 0
    for reg in scan_root(ROOT):
        feats = by.pop(reg.name, [])
        if not feats:
            continue
        out.extend(feats)
        ae = reg.rasters.get("ae") or []
        if not ae:
            continue
        n_ae += 1
        copies = li_to_ae(feats, ae, fp, fallback_crs=read_utm_json(reg.utm_json))
        n_copy += len(copies)
        out.extend(copies)
    for feats in by.values():
        out.extend(feats)
    dst.write_text(json.dumps(writer.to_feature_collection(out), ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"features": len(out), "copies": n_copy, "regions_with_ae": n_ae,
                      "unmatched_regions": len(by)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
