"""Shared DINOv2 loader for head training — same pin + path as inference.

Uses expds.gor_gate.load_backbone → import_dfine (vendor transformers 5.21.0
on Linux / Docker; env pin on Windows only if wheels match). Never Hub
from_pretrained. MODEL must be a local dir with config.json + model.safetensors.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_DEFAULT = HERE.parents[1]


def repo_root() -> Path:
    return Path(os.environ.get("REPO", str(REPO_DEFAULT))).resolve()


def probe_transformers(repo: Path | None = None) -> dict:
    """Import pinned transformers via import_dfine. Raises ImportError on failure."""
    root = Path(repo) if repo is not None else repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from expds.sp_dfine import import_dfine

    _cfg, _cls, info = import_dfine(root)
    return info


def preflight_model_dir(model: Path) -> list[str]:
    """Return list of human-readable problems (empty = OK)."""
    miss: list[str] = []
    m = Path(model)
    if not m.is_dir():
        miss.append(f"MODEL not a directory: {m}")
        return miss
    for name in ("config.json", "model.safetensors"):
        if not (m / name).is_file():
            miss.append(f"missing {m / name}")
    return miss


def load_dinov2_for_train(model_dir: str | Path, device: str, repo: Path | None = None):
    """Return (Dinov2Model.eval().to(device), info) matching gor_gate.load_backbone.

    info keys: hidden_size, n_prefix (1 + register tokens), transformers (import_dfine info).
    """
    root = Path(repo) if repo is not None else repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    model_dir = Path(model_dir)
    problems = preflight_model_dir(model_dir)
    if problems:
        raise FileNotFoundError("; ".join(problems))

    tr_info = probe_transformers(root)
    from expds.gor_gate import load_backbone

    # absolute MODEL path is supported by gor_gate._p
    m = load_backbone(
        {"backbone": str(model_dir.resolve()), "backbone_md5": None, "device": device},
        root=root,
        device=device,
    )
    cfg = m.config
    n_reg = int(getattr(cfg, "num_register_tokens", 0) or 0)
    info = {
        "hidden_size": int(cfg.hidden_size),
        "n_prefix": 1 + n_reg,
        "num_register_tokens": n_reg,
        "transformers": tr_info,
        "model_dir": str(model_dir.resolve()),
    }
    return m, info
