"""Проверка защиты от тихого отката облачного обучения на CPU и конфликта GPU-сессий Kaggle."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from expds.train import train

REPO_ROOT = Path(__file__).resolve().parents[1]


def _make_tiny_cache(tmp_path: Path) -> Path:
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "meta.json").write_text(
        '{"channels": 8, "classes": ["kurgany_tselye"], "tile_size": 32}',
        encoding="utf-8",
    )
    (cache / "index.csv").write_text(
        "shard,idx,region,family,positive\n"
        "0,0,reg_a,fam_a,1\n"
        "0,1,reg_b,fam_b,1\n",
        encoding="utf-8",
    )
    np.save(cache / "images_0000.npy", np.zeros((2, 8, 32, 32), dtype=np.float16))
    np.save(cache / "masks_0000.npy", np.ones((2, 32, 32), dtype=np.uint8))
    return cache


def test_train_refuses_silent_cpu_fallback_when_require_gpu(tmp_path: Path) -> None:
    cache = _make_tiny_cache(tmp_path)
    out_pt = tmp_path / "out.pt"
    with patch("torch.cuda.is_available", return_value=False):
        with pytest.raises(RuntimeError, match="Отказ от тихого отката на CPU"):
            train(
                str(cache),
                str(out_pt),
                epochs=1,
                batch=2,
                val_families=["fam_b"],
                pretrained=False,
                require_gpu=True,
            )


def test_train_refuses_cuda_device_when_cuda_unavailable(tmp_path: Path) -> None:
    cache = _make_tiny_cache(tmp_path)
    out_pt = tmp_path / "out.pt"
    with patch("torch.cuda.is_available", return_value=False):
        with pytest.raises(RuntimeError, match="Отказ от тихого отката на CPU"):
            train(
                str(cache),
                str(out_pt),
                epochs=1,
                batch=2,
                val_families=["fam_b"],
                pretrained=False,
                device="cuda:0",
            )


def test_orchestrator_enforces_gpu_and_single_session_guards() -> None:
    orch_txt = (REPO_ROOT / "scripts" / "kaggle_dual_t4_orchestrator.py").read_text(encoding="utf-8")
    assert '"enable_internet": "false"' in orch_txt
    assert "--batch 64 --lr 1.4e-3 --require-gpu" in orch_txt
    assert 'env["EXPDS_REQUIRE_GPU"] = "1"' in orch_txt
    assert "detected_gpus=0" in orch_txt
    assert "get_running_kernels_on_account" in orch_txt
