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
    "module_a": False, "module_b": False,
    "local_kernels": [3, 5, 7], "refinement_stages": [1, 2],
    "gate_mode": "deep_shallow", "gate_reduction": 4,
    "gate_init_probability": 1 / 3, "residual_init_std": 0.001,
    "decoder_fusion": "mlp", "decoder_upsampling": "bilinear",
    "upsample_channels": [64, 32],
    "freqfusion": False, "freqfusion_compressed_channels": 64,
    "freqfusion_lowpass_kernel": 5, "freqfusion_highpass_kernel": 3,
    "freqfusion_resample_groups": 4, "freqfusion_checkpoint": True,
    "freqfusion_chunk_size": 32,
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
    for key in ("module_a", "module_b", "freqfusion", "freqfusion_checkpoint"):
        if type(config[key]) is not bool:
            raise ValueError(f"{key} must be true or false.")
    if config["module_b"] and not config["module_a"]:
        raise ValueError("module_b selects module_a branches and requires module_a: true.")
    if config["decoder_fusion"] not in ("mlp", "progressive"):
        raise ValueError("decoder_fusion must be mlp or progressive.")
    if config["decoder_upsampling"] not in ("bilinear", "learned"):
        raise ValueError("decoder_upsampling must be bilinear or learned.")
    upsample_channels = config["upsample_channels"]
    if not isinstance(upsample_channels, list) or len(upsample_channels) != 2 or any(
        type(channels) is not int or channels < 1 for channels in upsample_channels
    ):
        raise ValueError("upsample_channels must contain two positive integer channel counts.")
    if (config["module_a"] or config["module_b"]) and (
        config["decoder_fusion"] != "mlp" or config["decoder_upsampling"] != "bilinear"
        or config["freqfusion"]
    ):
        raise ValueError(
            "Decoder redesign/FreqFusion experiments require module_a: false and module_b: false. "
            "Use the original mlp/bilinear decoder for the earlier A/B experiments."
        )
    for key in ("freqfusion_compressed_channels", "freqfusion_resample_groups", "freqfusion_chunk_size"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer.")
    if config["freqfusion_compressed_channels"] < 8:
        raise ValueError("freqfusion_compressed_channels must be at least eight.")
    for key in ("freqfusion_lowpass_kernel", "freqfusion_highpass_kernel"):
        kernel = config[key]
        if type(kernel) is not int or not 3 <= kernel <= 15 or kernel % 2 != 1:
            raise ValueError(f"{key} must be an odd integer from 3 to 15.")
    kernels = config["local_kernels"]
    if not isinstance(kernels, list) or not kernels or any(
        type(kernel) is not int or not 3 <= kernel <= 31 or kernel % 2 != 1
        for kernel in kernels
    ):
        raise ValueError("local_kernels must be a nonempty list of odd integers from 3 to 31.")
    stages = config["refinement_stages"]
    if not isinstance(stages, list) or not stages or any(
        type(stage) is not int or stage not in (1, 2) for stage in stages
    ) or len(set(stages)) != len(stages):
        raise ValueError("refinement_stages must contain unique stage numbers from 1 and 2.")
    if config["gate_mode"] not in ("shallow", "deep_shallow"):
        raise ValueError("gate_mode must be shallow or deep_shallow.")
    if type(config["gate_reduction"]) is not int or config["gate_reduction"] < 1:
        raise ValueError("gate_reduction must be a positive integer.")
    probability = config["gate_init_probability"]
    if type(probability) not in (int, float) or not math.isfinite(probability) or not 0 < probability < 1:
        raise ValueError("gate_init_probability must be finite and strictly between 0 and 1.")
    initial_std = config["residual_init_std"]
    if type(initial_std) not in (int, float) or not math.isfinite(initial_std) or initial_std < 0:
        raise ValueError("residual_init_std must be finite and nonnegative.")
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
