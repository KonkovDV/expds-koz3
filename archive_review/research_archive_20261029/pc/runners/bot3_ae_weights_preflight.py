"""Preflight Ae D-FINE HF weights: size>0, sha256 prefix, safetensors (or torch) load.

Usage:
  python bot3_ae_weights_preflight.py --dir DIR [--sha-prefix HEX] [--require-config] [--full-load]
  python bot3_ae_weights_preflight.py --file PATH [--sha-prefix HEX] [--full-load]

--full-load: require a real tensor load (for Docker pantum/n7q). Without it, host may
fall back to safetensors header validation if the safetensors package is missing.

Exit 0 on OK, 1 on load/size/sha failure, 2 on usage.
ASCII-only. No inline python -c from the PS runner.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def validate_safetensors_header(path: str) -> str:
    """Minimal format check without safetensors pkg (host fallback)."""
    with open(path, "rb") as f:
        hdr_len_b = f.read(8)
        if len(hdr_len_b) != 8:
            raise RuntimeError("safetensors_truncated_header")
        (hdr_len,) = struct.unpack("<Q", hdr_len_b)
        if hdr_len < 2 or hdr_len > 100 * 1024 * 1024:
            raise RuntimeError("safetensors_bad_header_len=%d" % hdr_len)
        raw = f.read(hdr_len)
        if len(raw) != hdr_len:
            raise RuntimeError("safetensors_short_header")
        meta = json.loads(raw.decode("utf-8"))
        if not isinstance(meta, dict) or not meta:
            raise RuntimeError("safetensors_empty_meta")
        # __metadata__ optional; tensor entries expected
        n_tensors = sum(1 for k in meta if k != "__metadata__")
        if n_tensors < 1:
            raise RuntimeError("safetensors_no_tensors_in_meta")
    return "safetensors_header n_tensors=%d" % n_tensors


def load_weights(path: str, full_load: bool) -> str:
    """Return loader name used. Raises on failure."""
    if path.endswith(".safetensors"):
        err = []
        try:
            from safetensors import safe_open

            with safe_open(path, framework="pt", device="cpu") as f:
                keys = list(f.keys())
                if not keys:
                    raise RuntimeError("safetensors has zero tensors")
                _ = f.get_tensor(keys[0])
            return "safetensors.safe_open"
        except Exception as e1:
            err.append(str(e1))
        try:
            from safetensors.torch import load_file

            sd = load_file(path, device="cpu")
            if not sd:
                raise RuntimeError("safetensors.torch.load_file empty")
            return "safetensors.torch.load_file"
        except Exception as e2:
            err.append(str(e2))
        if full_load:
            raise RuntimeError("safetensors_full_load_failed: %s" % " | ".join(err))
        # host fallback
        return validate_safetensors_header(path) + " (no safetensors pkg)"
    import torch

    torch.load(path, map_location="cpu", weights_only=False)
    return "torch.load"


def check_one(path: str, sha_prefix: str | None, full_load: bool) -> int:
    if not os.path.isfile(path):
        print("FAIL missing %s" % path)
        return 1
    size = os.path.getsize(path)
    if size <= 0:
        print("FAIL empty_file %s" % path)
        return 1
    digest = sha256_file(path)
    print("sha256=%s size=%d path=%s" % (digest, size, path))
    if sha_prefix:
        want = sha_prefix.lower().strip()
        if not digest.startswith(want):
            print("FAIL sha_prefix want=%s got=%s" % (want, digest))
            return 1
        print("sha_prefix OK %s" % want)
    try:
        how = load_weights(path, full_load=full_load)
        print("load OK via %s" % how)
    except Exception as e:
        print("FAIL load: %s" % e)
        return 1
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default=None, help="HF dir with model.safetensors [+ config.json]")
    ap.add_argument("--file", default=None, help="single weight file")
    ap.add_argument("--sha-prefix", default=None)
    ap.add_argument("--require-config", action="store_true")
    ap.add_argument("--full-load", action="store_true",
                    help="require real safetensors/torch load (Docker gate)")
    args = ap.parse_args()
    if bool(args.dir) == bool(args.file):
        print("usage: pass exactly one of --dir / --file", file=sys.stderr)
        return 2
    if args.file:
        return check_one(args.file, args.sha_prefix, args.full_load)
    st = os.path.join(args.dir, "model.safetensors")
    rc = check_one(st, args.sha_prefix, args.full_load)
    if rc != 0:
        return rc
    cfg = os.path.join(args.dir, "config.json")
    if args.require_config or os.path.isfile(cfg):
        if not os.path.isfile(cfg) or os.path.getsize(cfg) <= 0:
            print("FAIL missing_or_empty config.json in %s" % args.dir)
            return 1
        try:
            data = json.loads(open(cfg, encoding="utf-8").read())
            i2l = data.get("id2label") or {}
            print("config id2label_n=%d architectures=%s" % (
                len(i2l), data.get("architectures")))
            if not i2l:
                print("FAIL config.json missing id2label")
                return 1
        except Exception as e:
            print("FAIL config.json: %s" % e)
            return 1
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
