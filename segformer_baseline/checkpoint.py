"""Atomic checkpoints and epoch-boundary random-state restoration."""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch


SCHEMA_VERSION = 2


def atomic_torch_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_checkpoint(path: str | Path) -> dict:
    # Only load checkpoints from trusted sources: optimizer/RNG states require pickle.
    payload = torch.load(path, map_location="cpu", weights_only=False)
    required = {"model", "model_config", "normalization_mean", "normalization_std"}
    if not isinstance(payload, dict) or not required.issubset(payload):
        raise ValueError(f"Not a SegFormer baseline checkpoint: {path}")
    return payload


def preprocessing_from_checkpoint(checkpoint: dict) -> tuple[tuple[int, ...], np.ndarray, np.ndarray]:
    config = checkpoint.get("config", {})
    channels = tuple(config.get("input_channels", [3, 2, 1]))
    mean = np.asarray(checkpoint["normalization_mean"], dtype=np.float32)
    std = np.asarray(checkpoint["normalization_std"], dtype=np.float32)
    if len(channels) != checkpoint["model_config"].get("num_channels", 3):
        raise ValueError("Checkpoint input bands do not match its model architecture.")
    if mean.shape != (len(channels),) or std.shape != (len(channels),):
        raise ValueError("Checkpoint normalization statistics have an invalid shape.")
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any():
        raise ValueError("Checkpoint normalization statistics are invalid.")
    return channels, mean, std


def capture_rng(generators: dict[str, torch.Generator]) -> dict:
    return {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "loaders": {name: generator.get_state() for name, generator in generators.items()},
    }


def restore_rng(state: dict, generators: dict[str, torch.Generator]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state["cuda"]:
        if len(state["cuda"]) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all(state["cuda"])
        else:
            raise ValueError("CUDA device count changed; cannot restore all saved CUDA RNG states.")
    for name, generator in generators.items():
        generator.set_state(state["loaders"][name])


def make_checkpoint(model, optimizer, scheduler, epoch: int, best_iou: float,
                    best_epoch: int, config: dict, mean: np.ndarray, std: np.ndarray,
                    history: list[dict], manifest: dict, generators: dict,
                    environment: dict) -> dict:
    return {
        "schema_version": SCHEMA_VERSION, "model": model.state_dict(),
        "model_config": model.config.to_dict(), "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(), "epoch": epoch,
        "best_iou": best_iou, "best_epoch": best_epoch, "config": config,
        "normalization_mean": mean.tolist(), "normalization_std": std.tolist(),
        "history": history, "dataset_fingerprint": manifest["fingerprint"],
        "rng_state": capture_rng(generators), "environment": environment,
    }
