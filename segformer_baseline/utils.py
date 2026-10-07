"""Logging, experiment metadata and small serialization helpers."""

from __future__ import annotations

import csv
import importlib.metadata
import json
import logging
import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_device(name: str) -> torch.device:
    device = torch.device(name)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("This project supports cpu and cuda devices.")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. Check GPU PyTorch installation or explicitly use --device cpu.")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device does not exist: {name}")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.set_device(device)
    return device


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def setup_logger(path: Path) -> logging.Logger:
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("segformer")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (logging.FileHandler(path, encoding="utf-8"), logging.StreamHandler(sys.stdout)):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def write_history(directory: Path, history: list[dict]) -> None:
    write_csv(directory / "metrics.csv", history)
    temporary = directory / "metrics.jsonl.tmp"
    temporary.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in history), encoding="utf-8")
    os.replace(temporary, directory / "metrics.jsonl")


def collect_environment(repo_root: Path, device: torch.device) -> dict:
    def git(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", "-c", f"safe.directory={repo_root.as_posix()}", "-C", str(repo_root), *args],
                text=True, capture_output=True, check=True, timeout=10,
            )
            return result.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None

    packages = {}
    for name in ("torch", "transformers", "numpy", "tifffile", "tqdm", "PyYAML", "Pillow", "matplotlib"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "recorded_at": utc_now(), "python": sys.version, "python_executable": sys.executable,
        "platform": platform.platform(), "packages": packages,
        "torch_cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "cuda_device_count": torch.cuda.device_count(), "device": str(device),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "git_commit": git("rev-parse", "HEAD"), "git_status": git("status", "--porcelain"),
    }


def save_pip_freeze(path: Path) -> None:
    try:
        result = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as error:
        logging.getLogger("segformer").warning("Could not export pip freeze: %s", error)
        return
    if result.returncode == 0:
        path.write_text(result.stdout, encoding="utf-8")
    else:
        logging.getLogger("segformer").warning("Could not export pip freeze: %s", result.stderr.strip())
