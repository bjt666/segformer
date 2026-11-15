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

SHARED_SCALE_POLICY = (
    "Normalize original selected bands using original training pixels. Feed the unchanged normalized "
    "image to the native context path and its bilinear 2x interpolation (align_corners=False, "
    "antialias=False, half-up dimensions) to the main path, using ONE shared MiT encoder. "
    "Align and fuse selected context features into main features before ONE decoder. Keep masks "
    "unchanged; resize raw logits to the original grid before CE, Lovasz, softmax or argmax. "
    "Both paths are used during training, evaluation and prediction; no prediction ensemble."
)


def input_scale_policy(config: dict) -> str:
    return SHARED_SCALE_POLICY if config.get("shared_scale", "off") != "off" else INPUT_SCALE_POLICY


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
    """Keep the original normalized tensor for the optional shared context path."""
    architecture = getattr(getattr(model, "config", None), "local_refinement", {})
    shared = architecture.get("shared_scale", "off") != "off"
    context = images
    scale = model_input_scale(model)
    if shared and (scale != 2.0 or min(images.shape[-2:]) < 32):
        raise ValueError("Shared-scale forward requires input_scale 2.0 and native dimensions >= 32.")
    if scale != 1.0:
        size = scaled_size(images.shape[-2:], scale)
        if min(size) < 32:
            raise ValueError(f"Scaled input size {size} is too small for the four-stage MiT encoder (minimum 32).")
        images = F.interpolate(images, size=size, mode="bilinear", align_corners=False, antialias=False)
    if shared:
        return model(pixel_values=images, context_pixel_values=context, **kwargs)
    return model(pixel_values=images, **kwargs)


def logits_on_grid(logits: torch.Tensor, size) -> torch.Tensor:
    if logits.shape[-2:] == tuple(size):
        return logits
    return F.interpolate(logits, size=size, mode="bilinear", align_corners=False)
