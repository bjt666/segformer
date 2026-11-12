"""Lightweight UnravelNet-inspired residual adapter, used only after MiT F2.

Design reference: UnravelNet (ISPRS JPRS 2026), Wei Lu et al.
https://github.com/AeroVILab-AHU/LEGNet/blob/main/mmrotate/models/backbones/unravelnet.py
Upstream copyright (c) 2025 Wei Lu; MIT notice: THIRD_PARTY_NOTICES.md.

This is an adaptation, not a reproduction of the published backbone. It uses
small quality-path widths, GroupNorm, normalized fixed Gaussian filtering,
shape-safe directional kernels, and a small residual projection. No MMCV,
antialiased-cnns, extra pretrained weights, or auxiliary labels are required.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F
from transformers.modeling_outputs import BaseModelOutput


def _norm(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(math.gcd(channels, max(1, channels // 8)), channels)


def _conv(channels: int, output: int, kernel: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(channels, output, kernel, padding=kernel // 2, bias=False),
        _norm(output), nn.GELU(),
    )


class _GaussianFilter(nn.Module):
    """Normalized, nontrainable 3x3 binomial Gaussian; preserves constants."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        vector = torch.tensor([1.0, 2.0, 1.0])
        kernel = (vector[:, None] * vector[None, :]) / 16.0
        self.register_buffer("kernel", kernel[None, None].repeat(channels, 1, 1, 1))

    def forward(self, features: torch.Tensor, stride: int = 1) -> torch.Tensor:
        padded = F.pad(features, (1, 1, 1, 1), mode="replicate")
        return F.conv2d(padded, self.kernel.to(dtype=features.dtype),
                        stride=stride, groups=features.shape[1])


class _DirectionalContext(nn.Module):
    """Horizontal, vertical and two diagonal directions on a pooled map.

    Diagonal kernels have only K trainable coefficients per channel. Their
    dense convolution realization supports rectangular and odd image sizes.
    """

    def __init__(self, channels: int, kernel: int) -> None:
        super().__init__()
        self.blur = _GaussianFilter(channels)
        self.horizontal = nn.Conv2d(channels, channels, (1, kernel),
                                    padding=(0, kernel // 2), groups=channels, bias=False)
        self.vertical = nn.Conv2d(channels, channels, (kernel, 1),
                                  padding=(kernel // 2, 0), groups=channels, bias=False)
        self.diagonal = nn.Parameter(torch.empty(2, channels, kernel))
        nn.init.normal_(self.diagonal, std=kernel ** -0.5)
        self.norm = _norm(channels)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        pooled = self.blur(F.max_pool2d(features, 3, stride=1, padding=1), stride=3)
        diagonal = torch.diag_embed(self.diagonal[0]).unsqueeze(1)
        opposite = torch.diag_embed(self.diagonal[1]).flip(-1).unsqueeze(1)
        padding = self.diagonal.shape[-1] // 2
        context = self.horizontal(pooled) + self.vertical(pooled)
        context = context + F.conv2d(pooled, diagonal, padding=padding, groups=features.shape[1])
        context = context + F.conv2d(pooled, opposite, padding=padding, groups=features.shape[1])
        gate = F.interpolate(self.norm(context).sigmoid(), size=features.shape[-2:],
                             mode="bilinear", align_corners=False)
        return features * gate


class _QualityContext(nn.Module):
    """Gaussian-related feature prior with spatial and channel interaction.

    This path does not estimate noise or use foreground/edge ground truth.
    Its benefit on weak algae and water false positives must be evaluated.
    """

    def __init__(self, channels: int, width: int) -> None:
        super().__init__()
        self.gaussian = _GaussianFilter(channels)
        self.prior = nn.Sequential(
            _conv(channels, width // 2), _conv(width // 2, width, 3),
            _conv(width, width, 3), nn.Conv2d(width, channels, 1, bias=False), _norm(channels),
        )
        self.interaction = _conv(channels, channels, 3)
        self.channel_gate = nn.Conv1d(1, 1, 3, padding=1, bias=False)
        self.norm = _norm(channels)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        prior = self.prior(features + self.gaussian(features))
        interaction = self.interaction(features + features * prior)
        pooled = interaction.mean(dim=(-2, -1)).unsqueeze(1)
        gate = self.channel_gate(pooled).sigmoid().squeeze(1)[..., None, None]
        return self.norm(features + interaction * gate)


class F2UnravelAdapter(nn.Module):
    """Same-shape F2 -> F2 + scale * correction, with four channel groups."""

    def __init__(self, channels: int, width: int = 128, kernel: int = 7,
                 quality_width: int = 64, residual_scale: float = 0.1) -> None:
        super().__init__()
        if channels < 1 or width < 16 or width % 4:
            raise ValueError("Unravel width must be at least 16 and divisible by four.")
        if kernel < 3 or kernel % 2 != 1:
            raise ValueError("Unravel directional kernel must be odd and at least three.")
        if quality_width < 16 or quality_width % 2:
            raise ValueError("Unravel quality width must be even and at least 16.")
        if not math.isfinite(residual_scale) or not 0 < residual_scale <= 1:
            raise ValueError("Unravel initial residual scale must be in (0, 1].")
        group = width // 4
        self.project = _conv(channels, width)
        self.point_gate = nn.Sequential(
            _conv(group, 4 * group), nn.Conv2d(4 * group, group, 1, bias=False), nn.Sigmoid(),
        )
        self.local = _conv(group, group, 3)
        self.directional = _DirectionalContext(group, kernel)
        self.quality = _QualityContext(group, quality_width)
        self.mix = nn.Sequential(_conv(width, 2 * width), nn.Conv2d(2 * width, width, 1, bias=False))
        self.output_projection = nn.Conv2d(width, channels, 1, bias=False)
        nn.init.normal_(self.output_projection.weight, std=0.001)
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale)))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        point, local, directional, quality = self.project(features).chunk(4, dim=1)
        point = point + point * self.point_gate(point)
        refined = torch.cat((point, self.local(local), self.directional(directional),
                             self.quality(quality)), dim=1)
        correction = self.output_projection(self.mix(refined))
        return features + self.residual_scale * correction


class UnravelEncoder(nn.Module):
    """Preserve MiT tensor names and insert one adapter after stage-2 norm.

    The adapted F2 is both saved for the decoder and passed into stage 3.
    F1 remains unchanged; F3/F4 are recomputed from the adapted representation.
    The ordinary HF encoder output API is retained for either decoder.
    """

    def __init__(self, original_encoder, model_config, architecture: dict) -> None:
        super().__init__()
        if len(model_config.hidden_sizes) != 4 or not model_config.reshape_last_stage:
            raise ValueError("F2 Unravel requires four native 4D MiT feature stages.")
        self.config = model_config
        self.patch_embeddings = original_encoder.patch_embeddings
        self.block = original_encoder.block
        self.layer_norm = original_encoder.layer_norm
        self.unravel = F2UnravelAdapter(
            model_config.hidden_sizes[1], width=architecture["unravel_channels"],
            kernel=architecture["unravel_kernel"], quality_width=architecture["unravel_quality_channels"],
            residual_scale=architecture["unravel_residual_scale"],
        )

    def forward(self, pixel_values: torch.Tensor, output_attentions: bool = False,
                output_hidden_states: bool = False, return_dict: bool = True):
        hidden_states = pixel_values
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        batch = pixel_values.shape[0]
        for index, (embedding, blocks, norm) in enumerate(
            zip(self.patch_embeddings, self.block, self.layer_norm)
        ):
            hidden_states, height, width = embedding(hidden_states)
            for block in blocks:
                outputs = block(hidden_states, height, width, output_attentions=output_attentions)
                hidden_states = outputs[0]
                if output_attentions:
                    all_self_attentions += (outputs[1],)
            hidden_states = norm(hidden_states)
            hidden_states = hidden_states.reshape(batch, height, width, -1).permute(0, 3, 1, 2).contiguous()
            if index == 1:
                hidden_states = self.unravel(hidden_states)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
        if not return_dict:
            return tuple(item for item in (hidden_states, all_hidden_states, all_self_attentions) if item is not None)
        return BaseModelOutput(last_hidden_state=hidden_states, hidden_states=all_hidden_states,
                               attentions=all_self_attentions)
