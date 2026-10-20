#!/usr/bin/env python3
"""Сборка Rust-ускорителя expds_core.

Для разработки (Windows или Linux, в активном виртуальном окружении):
    python scripts/build_rust.py dev
    → maturin develop --release: модуль ставится в текущий Python.

Для архива на платформу (контейнер Linux x86_64, Python ≥ 3.12):
    python scripts/build_rust.py linux
    → собирает manylinux-колесо и кладёт expds_core*.so в inference/lib/.
    Нужен Linux или WSL2 (на Windows) либо Docker; варианты ниже.

Без Rust всё работает на NumPy-версиях тех же функций, только медленнее.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CRATE = ROOT / "rust" / "expds_core"
LIB = ROOT / "inference" / "lib"
WHEELS = ROOT / "rust" / "target" / "wheels"


def run(cmd: list[str], cwd: Path | None = None, env: dict | None = None) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True, env=env)


def dev() -> None:
    """Сборка и установка модуля в текущее окружение, под процессор этой машины."""
    import os

    run([sys.executable, "-m", "pip", "install", "maturin>=1.7,<2"])
    env = {**os.environ, "RUSTFLAGS": "-C target-cpu=native"}
    run(["maturin", "develop", "--release", "-m", str(CRATE / "Cargo.toml")], env=env)
    run([sys.executable, "-c", "import expds_core; print('expds_core', expds_core.version())"])


def linux(docker: bool) -> None:
    """manylinux-колесо → .so в inference/lib (совместимо с Python ≥ 3.12, abi3).

    target-cpu=x86-64-v3 — AVX2 и FMA, базовый набор любого сервера с RTX 4090.
    native этой машины (Raptor Lake) в колесо платформы не попадает: на другом
    процессоре такой бинарник может не запуститься.
    """
    import os

    WHEELS.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "RUSTFLAGS": "-C target-cpu=x86-64-v3"}
    if docker:
        # Официальный образ maturin: сборка в manylinux_2_28 без установки Rust на хост.
        run(["docker", "run", "--rm", "-e", "RUSTFLAGS=-C target-cpu=x86-64-v3",
             "-v", f"{ROOT}:/io", "ghcr.io/pyo3/maturin", "build", "--release",
             "-m", "rust/expds_core/Cargo.toml", "--compatibility", "manylinux_2_28", "-o", "rust/target/wheels"])
    else:
        run([sys.executable, "-m", "pip", "install", "maturin>=1.7,<2"])
        run(["maturin", "build", "--release", "-m", str(CRATE / "Cargo.toml"), "-o", str(WHEELS)], env=env)
    wheels = sorted(WHEELS.glob("expds_core-*-linux*.whl")) + sorted(WHEELS.glob("expds_core-*manylinux*.whl"))
    if not wheels:
        sys.exit("колесо Linux не найдено в " + str(WHEELS))
    wheel = max(wheels, key=lambda p: p.stat().st_mtime)
    LIB.mkdir(parents=True, exist_ok=True)
    for old in LIB.glob("expds_core*.so"):
        old.unlink()
    with zipfile.ZipFile(wheel) as z:
        for name in z.namelist():
            if name.endswith(".so") and Path(name).name.startswith("expds_core"):
                target = LIB / Path(name).name
                with z.open(name) as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                print("готово:", target.relative_to(ROOT), f"({target.stat().st_size // 1024} КБ) из", wheel.name)
                return
    sys.exit("в колесе нет .so")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target", choices=["dev", "linux"])
    ap.add_argument("--docker", action="store_true", help="собрать в контейнере ghcr.io/pyo3/maturin")
    args = ap.parse_args()
    dev() if args.target == "dev" else linux(args.docker)


if __name__ == "__main__":
    main()
