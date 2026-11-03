"""Rebuild КОЗ №3 train tree from Kaggle xfer shards + GT dataset (bot-6).

Shards (bot-1):
  dmitrijkv/koz3-xfer-s000  — small_files.tar.zst (+ manifest; labels/UTM/extra)
  dmitrijkv/koz3-xfer-s001 … s009 — large blobs f####.tif / f####.laz;
      each shard has a manifest mapping blob → original relative path
  dmitrijkv/koz3-gt-train-v2 — consolidated GT (optional overlay for PIXGBM;
      per-region ObjectPoly/FindsPoints already live in s000 small tar)

Until shard manifests for s001–s009 are known offline, attach ALL shards;
at runtime only materialize paths whose top folder is geo 200–217
(PIXGBM GEO_REGIONS), unless PIXGBM_REBUILD_ALL=1.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tarfile
from pathlib import Path
from typing import Any, Iterator

GEO_PREFIXES = tuple(f"{n:03d}_" for n in range(200, 218))
BLOB_RE = re.compile(r"^f\d{4}(\.[A-Za-z0-9]+)?$")


def _is_geo_rel(rel: str) -> bool:
    rel = rel.replace("\\", "/").lstrip("./")
    top = rel.split("/", 1)[0]
    return any(top.startswith(p) for p in GEO_PREFIXES)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def find_input_datasets(input_root: Path | None = None) -> dict[str, Path]:
    """Map short name → /kaggle/input/<slug> directory."""
    root = input_root or Path("/kaggle/input")
    out: dict[str, Path] = {}
    if not root.is_dir():
        return out
    for p in sorted(root.iterdir()):
        if not p.is_dir():
            continue
        name = p.name.lower()
        out[name] = p
    return out


def find_xfer_shards(datasets: dict[str, Path]) -> list[tuple[int, Path]]:
    found: list[tuple[int, Path]] = []
    for name, path in datasets.items():
        m = re.search(r"koz3-xfer-s(\d{3})$", name)
        if m:
            found.append((int(m.group(1)), path))
    return sorted(found)


def find_gt_dataset(datasets: dict[str, Path]) -> Path | None:
    for key in ("koz3-gt-train-v2", "koz3-gt-train"):
        if key in datasets:
            return datasets[key]
    for name, path in datasets.items():
        if "gt-train" in name or "gt_train" in name:
            return path
    return None


def find_manifest(shard: Path) -> Path | None:
    for cand in (
        shard / "manifest.json",
        shard / f"{shard.name}_manifest.json",
        shard / "koz3-xfer-manifest.json",
    ):
        if cand.is_file():
            return cand
    # any *manifest*.json at top level
    hits = sorted(shard.glob("*manifest*.json"))
    return hits[0] if hits else None


def find_small_tar(shard: Path) -> Path | None:
    for cand in (
        shard / "small_files.tar.zst",
        shard / "small_files.tar.zstd",
        shard / "small_files.tar",
    ):
        if cand.is_file():
            return cand
    hits = list(shard.glob("small_files.tar*"))
    return hits[0] if hits else None


def _is_blob_id(s: str) -> bool:
    name = Path(str(s)).name
    return bool(BLOB_RE.match(name) or re.match(r"^f\d{4}$", name))


def iter_blob_mappings(manifest: Any) -> Iterator[tuple[str, str]]:
    """Yield (blob_basename, relative_path) from flexible manifest shapes."""
    if not isinstance(manifest, dict):
        return

    def emit(blob: str, rel: str):
        rel = rel.replace("\\", "/").lstrip("./")
        blob = Path(str(blob)).name
        if rel and blob:
            yield blob, rel

    def from_members(members: dict):
        for k, v in members.items():
            if isinstance(v, str):
                if _is_blob_id(k):
                    yield from emit(k, v)
                elif _is_blob_id(v):
                    yield from emit(v, k)
            elif isinstance(v, dict):
                rel = v.get("rel") or v.get("path") or v.get("orig") or v.get("original")
                blob = v.get("blob") or v.get("name") or v.get("file") or v.get("id") or v.get("key")
                if _is_blob_id(k) and rel:
                    yield from emit(k, str(rel))
                elif blob and rel:
                    yield from emit(str(blob), str(rel))
                elif blob and "/" in str(k):
                    yield from emit(str(blob), str(k))

    for key in ("files", "large", "blobs", "members", "map", "mapping"):
        block = manifest.get(key)
        if isinstance(block, dict):
            if isinstance(block.get("members"), dict):
                yield from from_members(block["members"])
            else:
                yield from from_members(block)

    top = {k: v for k, v in manifest.items() if _is_blob_id(str(k))}
    if top:
        yield from from_members(top)


def extract_small_tar(tar_path: Path, dest: Path) -> int:
    """Extract small_files tar(.zst) into dest. Returns member count."""
    dest.mkdir(parents=True, exist_ok=True)
    name = tar_path.name
    if name.endswith(".zst") or name.endswith(".zstd"):
        # Prefer zstd CLI (Kaggle images usually have it); fall back to pip zstandard.
        tmp_tar = dest.parent / "_small_files.tar"
        if tmp_tar.exists():
            tmp_tar.unlink()
        try:
            subprocess.check_call(["zstd", "-d", "-f", "-o", str(tmp_tar), str(tar_path)])
        except (FileNotFoundError, subprocess.CalledProcessError):
            try:
                import zstandard as zstd  # type: ignore
            except ImportError:
                subprocess.check_call([sys_executable(), "-m", "pip", "install", "-q", "zstandard"])
                import zstandard as zstd  # type: ignore
            dctx = zstd.ZstdDecompressor()
            with tar_path.open("rb") as src, tmp_tar.open("wb") as out:
                dctx.copy_stream(src, out)
        path_for_tar = tmp_tar
    else:
        path_for_tar = tar_path

    n = 0
    with tarfile.open(path_for_tar, mode="r:*") as tf:
        for m in tf.getmembers():
            if not m.isfile():
                continue
            rel = m.name.lstrip("./")
            if not _want_rel(rel):
                continue
            tf.extract(m, path=dest, filter="data")
            n += 1
    if path_for_tar != tar_path and path_for_tar.exists():
        path_for_tar.unlink(missing_ok=True)
    return n


def sys_executable() -> str:
    import sys

    return sys.executable


def _want_rel(rel: str) -> bool:
    if os.environ.get("PIXGBM_REBUILD_ALL", "").strip() in ("1", "true", "yes"):
        return True
    return _is_geo_rel(rel)


def resolve_blob(shard: Path, blob: str) -> Path | None:
    """Find f#### or f####.ext inside shard directory."""
    b = Path(blob).name
    direct = shard / b
    if direct.is_file():
        return direct
    stem = b.split(".")[0]
    # try exact stem with any suffix
    hits = list(shard.glob(stem + ".*")) + list(shard.glob(stem))
    for h in hits:
        if h.is_file():
            return h
    # nested one level
    for h in shard.glob("*/" + stem + ".*"):
        if h.is_file():
            return h
    return None


def link_or_copy(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return "exists"
    try:
        os.symlink(src, dst)
        return "symlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def materialize_shard_blobs(shard: Path, tree: Path, manifest_path: Path | None) -> dict:
    stats = {"shard": shard.name, "linked": 0, "copied": 0, "exists": 0, "missing_blob": 0, "skipped_non_geo": 0, "mapped": 0}
    if manifest_path is None:
        stats["error"] = "no_manifest"
        return stats
    man = _load_json(manifest_path)
    mappings = list(iter_blob_mappings(man))
    stats["mapped"] = len(mappings)
    if not mappings:
        stats["error"] = "no_blob_mappings_in_manifest"
        return stats
    for blob, rel in mappings:
        if not _want_rel(rel):
            stats["skipped_non_geo"] += 1
            continue
        src = resolve_blob(shard, blob)
        if src is None:
            # blob id without extension — try as-is
            src = resolve_blob(shard, blob.split(".")[0])
        if src is None:
            stats["missing_blob"] += 1
            continue
        # preserve extension from actual file if rel has none
        rel_path = rel.replace("\\", "/")
        if not Path(rel_path).suffix and src.suffix:
            rel_path = rel_path + src.suffix
        action = link_or_copy(src, tree / rel_path)
        stats[action] = stats.get(action, 0) + 1
    return stats


def overlay_gt(gt_root: Path, tree: Path) -> dict:
    """Best-effort: copy geojson/annotations from gt dataset into tree if missing."""
    stats = {"gt": gt_root.name, "copied": 0, "skipped": 0}
    # If dataset is a single geojson — leave for other pipelines; PIXGBM uses per-region files.
    for p in gt_root.rglob("*"):
        if not p.is_file():
            continue
        try:
            rel = p.relative_to(gt_root).as_posix()
        except ValueError:
            continue
        if not _want_rel(rel) and not rel.endswith(".geojson"):
            # allow geojson overlays only under geo regions
            if not any(part.startswith(GEO_PREFIXES) for part in Path(rel).parts):
                stats["skipped"] += 1
                continue
        if not _want_rel(rel):
            stats["skipped"] += 1
            continue
        dst = tree / rel
        if dst.exists():
            stats["skipped"] += 1
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dst)
        stats["copied"] += 1
    return stats


def rebuild_train_tree(
    dest: Path | None = None,
    input_root: Path | None = None,
) -> dict:
    """Build train root under dest; return stats dict including 'root' path."""
    dest = Path(dest or os.environ.get("PIXGBM_TREE", "/kaggle/working/train-expds-3"))
    datasets = find_input_datasets(input_root)
    shards = find_xfer_shards(datasets)
    report: dict[str, Any] = {
        "dest": str(dest),
        "shards_found": [f"s{idx:03d}:{path.name}" for idx, path in shards],
        "steps": [],
    }
    if not shards:
        raise FileNotFoundError(
            "No koz3-xfer-sNNN datasets under /kaggle/input. "
            "Attach dmitrijkv/koz3-xfer-s000 … s009."
        )

    dest.mkdir(parents=True, exist_ok=True)
    # s000 small tar first
    for idx, path in shards:
        if idx != 0:
            continue
        tar = find_small_tar(path)
        if tar is None:
            report["steps"].append({"s000": path.name, "error": "small_files.tar* missing"})
        else:
            n = extract_small_tar(tar, dest)
            report["steps"].append({"s000_extract": tar.name, "members_geo": n})

    # large shards (+ s000 if it also has blob maps)
    for idx, path in shards:
        man = find_manifest(path)
        if idx == 0 and man is None:
            # optional: shared copy path name
            continue
        st = materialize_shard_blobs(path, dest, man)
        report["steps"].append(st)

    gt = find_gt_dataset(datasets)
    if gt is not None:
        report["steps"].append(overlay_gt(gt, dest))
    else:
        report["steps"].append({"gt": None, "note": "koz3-gt-train-v2 not attached"})

    # sanity: count geo region dirs (200–217) and any region dirs
    geo_dirs = sorted(
        p.name for p in dest.iterdir() if p.is_dir() and any(p.name.startswith(pref) for pref in GEO_PREFIXES)
    )
    any_region_dirs = sorted(
        p.name for p in dest.iterdir() if p.is_dir() and len(p.name) >= 4 and p.name[:3].isdigit()
    )
    report["geo_region_dirs"] = geo_dirs
    report["any_region_dirs_n"] = len(any_region_dirs)
    report["any_region_dirs_sample"] = any_region_dirs[:40]
    report["root"] = str(dest)
    rebuild_all = os.environ.get("PIXGBM_REBUILD_ALL", "").strip() in ("1", "true", "yes")
    if not geo_dirs:
        if rebuild_all and any_region_dirs:
            # honest10 / full tree: geos 1–33 rasters live outside 200–217; 200–217 often hollow
            report["note"] = (
                "PIXGBM_REBUILD_ALL=1: no geo 200–217 dirs with content gate skipped; "
                f"using {len(any_region_dirs)} region dirs"
            )
        else:
            raise FileNotFoundError(
                f"Rebuild produced no geo 200–217 dirs under {dest}. "
                "Check shard manifests / small_files extract."
            )
    return report


if __name__ == "__main__":
    import pprint

    pprint.pp(rebuild_train_tree())
