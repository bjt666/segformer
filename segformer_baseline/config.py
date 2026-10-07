"""Configuration loading and validation; all paths are relative to the cwd."""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

import yaml


DEFAULTS: dict[str, Any] = {
    "data_root": "dataset", "output_dir": None, "epochs": 100,
    "batch_size": 8, "num_workers": 4, "seed": 42, "device": "cuda",
    "pretrained": True, "pretrained_model": "nvidia/mit-b0",
    "input_channels": [3, 2, 1], "input_bands": ["NIR", "R", "G"],
    "encoder_lr": 6e-5, "decoder_lr": 6e-4, "weight_decay": 0.01,
    "loss": "cross_entropy", "log_every": 20, "visualization_samples": 8,
}


def read_config(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    with Path(path).open(encoding="utf-8") as handle:
        values = yaml.safe_load(handle)
    if not isinstance(values, dict):
        raise ValueError("The configuration must be a YAML mapping.")
    unknown = set(values) - DEFAULTS.keys()
    if unknown:
        raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
    return values


def default_config() -> dict[str, Any]:
    return copy.deepcopy(DEFAULTS)


def validate_config(config: dict[str, Any]) -> None:
    for key in ("epochs", "batch_size", "log_every"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer.")
    for key in ("num_workers", "visualization_samples", "seed"):
        if type(config[key]) is not int or config[key] < 0:
            raise ValueError(f"{key} must be a nonnegative integer.")
    for key in ("encoder_lr", "decoder_lr"):
        if type(config[key]) not in (int, float) or not math.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f"{key} must be positive.")
    if type(config["weight_decay"]) not in (int, float) or not math.isfinite(config["weight_decay"]) or config["weight_decay"] < 0:
        raise ValueError("weight_decay must be nonnegative.")
    if type(config["pretrained"]) is not bool:
        raise ValueError("pretrained must be true or false.")
    for key in ("data_root", "device", "pretrained_model"):
        if not isinstance(config[key], str) or not config[key]:
            raise ValueError(f"{key} must be a nonempty string.")
    if config["output_dir"] is not None and (not isinstance(config["output_dir"], str) or not config["output_dir"]):
        raise ValueError("output_dir must be null or a nonempty string.")
    channels = config["input_channels"]
    if not isinstance(channels, list) or not channels or any(
        type(channel) is not int or channel not in range(4) for channel in channels
    ) or len(set(channels)) != len(channels):
        raise ValueError("input_channels must contain unique indices from 0 to 3.")
    bands = config["input_bands"]
    if not isinstance(bands, list) or len(bands) != len(channels) or any(
        not isinstance(band, str) or not band for band in bands
    ):
        raise ValueError("input_bands must name each selected channel.")
    if config["loss"] != "cross_entropy":
        raise ValueError("This baseline implements unweighted cross_entropy only.")


def save_config(config: dict[str, Any], path: Path) -> None:
    path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
