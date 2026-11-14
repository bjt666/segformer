"""Single-scale image interpolation without changing annotation/evaluation grids."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


INPUT_SCALE_POLICY = (
    "Normalize original selected bands using original training pixels, then bilinear resize "
    "images with align_corners=False and antialias=False. Round scaled dimensions half up. "
    "Keep masks unchanged. Bilinear resize raw logits to the original mask/image grid "
    "before CE, Lovasz, softmax or argmax. One input scale, not multi-scale inference."
)


def validate_input_scale(value) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= 4:
        raise ValueError("input_scale must be finite and in (0, 4].")
    return float(value)


def scaled_size(size, scale: float) -> tuple[int, int]:
    scale = validate_input_scale(scale)
    return tuple(max(1, int(math.floor(int(dimension) * scale + 0.5))) for dimension in size)


def model_input_scale(model) -> float:
    return validate_input_scale(getattr(getattr(model, "config", None), "input_scale", 1.0))


def checkpoint_input_scale(checkpoint: dict) -> float:
    """Reject contradictory metadata; legacy checkpoints default to scale one."""
    descriptions = [(name, container["input_scale"]) for name, container in (
        ("input_scale", checkpoint), ("config.input_scale", checkpoint.get("config", {})),
        ("model_config.input_scale", checkpoint.get("model_config", {})),
    ) if "input_scale" in container]
    if not descriptions:
        return 1.0
    name, value = descriptions[0]
    scale = validate_input_scale(value)
    for other_name, other_value in descriptions[1:]:
        if validate_input_scale(other_value) != scale:
            raise ValueError(f"Checkpoint input_scale metadata disagree: {name} vs {other_name}.")
    return scale


def forward_with_input_scale(model, images: torch.Tensor, **kwargs):
    """All CLI forward paths share this function; the model itself stays intact."""
    scale = model_input_scale(model)
    if scale != 1.0:
        size = scaled_size(images.shape[-2:], scale)
        if min(size) < 32:
            raise ValueError(f"Scaled input size {size} is too small for the four-stage MiT encoder (minimum 32).")
        images = F.interpolate(images, size=size, mode="bilinear", align_corners=False, antialias=False)
    return model(pixel_values=images, **kwargs)


def logits_on_grid(logits: torch.Tensor, size) -> torch.Tensor:
    if logits.shape[-2:] == tuple(size):
        return logits
    return F.interpolate(logits, size=size, mode="bilinear", align_corners=False)
