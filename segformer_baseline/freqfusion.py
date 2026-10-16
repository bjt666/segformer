"""PyTorch implementation of the FreqFusion mechanism for SegFormer.

Reference: Chen et al., Frequency-Aware Feature Fusion for Dense Image
Prediction, TPAMI 2024, doi:10.1109/TPAMI.2024.3449959 (arXiv:2408.12879).
This implementation follows the published adaptive low/high-pass filtering,
enhanced initial fusion, and local-similarity-guided resampling mechanism.
It is a project adaptation, not a numerical reproduction of the author's
SegFormer experiments or a vendored copy of the reference implementation.

Dynamic reassembly uses native PyTorch, source-resolution patches, channel
chunks, and optional activation checkpointing. Replicate padding handles
small feature maps. Adjacent MiT stages may have sizes 2*n or 2*n-1; missing
bottom/right subpixels are padded for the calculation, then cropped away.
The grouped sampler uses an equivalent high-resolution identity-grid form.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def _reassemble_chunk(
    features: torch.Tensor, weights: torch.Tensor, kernel: int, scale: int
) -> torch.Tensor:
    """Apply per-position kernels without lifting patches to output resolution."""
    batch, channels, height, width = features.shape
    radius = kernel // 2
    patches = F.unfold(
        F.pad(features, (radius, radius, radius, radius), mode="replicate"), kernel
    ).reshape(batch, channels, kernel * kernel, height, width)
    if scale == 1:
        return torch.einsum("bckhw,bkhw->bchw", patches, weights)

    # Four separate subpixel kernels share the same source-resolution patches.
    # B,C,4,H,W -> pixel_shuffle -> B,C,2H,2W, in row-major subpixel order.
    subpixels = [
        torch.einsum("bckhw,bkhw->bchw", patches, weights[:, :, y::2, x::2])
        for y in range(2) for x in range(2)
    ]
    packed = torch.stack(subpixels, dim=2).reshape(batch, channels * 4, height, width)
    return F.pixel_shuffle(packed, 2)


def _local_similarity(features: torch.Tensor) -> torch.Tensor:
    """Eight cosine similarities in a 3x3 neighborhood with dilation two."""
    batch, channels, height, width = features.shape
    normalized = F.normalize(features, dim=1, eps=1e-6)
    neighbors = F.unfold(normalized, kernel_size=3, dilation=2, padding=2)
    neighbors = neighbors.reshape(batch, channels, 9, height, width)
    similarity = (normalized.unsqueeze(2) * neighbors).sum(dim=1)
    return torch.cat((similarity[:, :4], similarity[:, 5:]), dim=1)


class _SimilaritySampler(nn.Module):
    """Predict grouped residual sampling offsets from both feature levels."""

    def __init__(self, compressed_channels: int, groups: int) -> None:
        super().__init__()
        self.groups = groups
        norm_groups = math.gcd(compressed_channels, max(1, compressed_channels // 8))
        self.high_norm = nn.GroupNorm(norm_groups, compressed_channels)
        self.low_norm = nn.GroupNorm(norm_groups, compressed_channels)
        inputs = compressed_channels + 8
        self.high_direction = nn.Conv2d(inputs, 2 * groups, 3, padding=1)
        self.low_direction = nn.Conv2d(inputs, 8 * groups, 3, padding=1)
        self.high_scope = nn.Conv2d(inputs, 2 * groups, 3, padding=1)
        self.low_scope = nn.Conv2d(inputs, 8 * groups, 3, padding=1)
        for layer in (self.high_direction, self.low_direction):
            nn.init.normal_(layer.weight, std=0.001)
            nn.init.zeros_(layer.bias)
        for layer in (self.high_scope, self.low_scope):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(
        self, high: torch.Tensor, low: torch.Tensor, features: torch.Tensor
    ) -> torch.Tensor:
        batch, channels, height, width = features.shape
        high = self.high_norm(high)
        low = self.low_norm(low)
        high_context = torch.cat((high, _local_similarity(high)), dim=1)
        low_context = torch.cat((low, _local_similarity(low)), dim=1)
        low_direction = F.pixel_shuffle(self.low_direction(low_context), 2)
        low_scope = F.pixel_shuffle(self.low_scope(low_context), 2)
        direction = self.high_direction(high_context) + low_direction[:, :, :height, :width]
        scope = self.high_scope(high_context) + low_scope[:, :, :height, :width]
        # Direction is expressed in low-resolution pixels. One low pixel spans
        # two pixels of the already upsampled features that are being sampled.
        displacement = (2.0 * direction * scope.sigmoid()).reshape(
            batch, 2, self.groups, height, width
        ).permute(0, 2, 3, 4, 1)
        y, x = torch.meshgrid(
            torch.arange(height, device=features.device, dtype=features.dtype),
            torch.arange(width, device=features.device, dtype=features.dtype),
            indexing="ij",
        )
        position = torch.stack((x + 0.5, y + 0.5), dim=-1)[None, None]
        normalizer = features.new_tensor((width, height))
        grid = 2.0 * (position + displacement) / normalizer - 1.0
        sampled = F.grid_sample(
            features.reshape(batch * self.groups, channels // self.groups, height, width),
            grid.reshape(batch * self.groups, height, width, 2),
            mode="bilinear", padding_mode="border", align_corners=False,
        )
        return sampled.reshape(batch, channels, height, width)


class FreqFusion(nn.Module):
    """Return enhanced high features and aligned low features, without merging.

    All three mechanisms are active: adaptive low-pass upsampling, adaptive
    high-pass residual enhancement, and grouped similarity-guided resampling.
    The caller chooses concatenation (original MLP pathway) or addition
    (optional progressive decoder). Channels of the two inputs may differ.
    """

    def __init__(
        self, high_channels: int, low_channels: int, *, compressed_channels: int = 64,
        lowpass_kernel: int = 5, highpass_kernel: int = 3, resample_groups: int = 4,
        checkpoint_reassembly: bool = True, chunk_size: int = 32,
    ) -> None:
        super().__init__()
        for name, value in (
            ("high_channels", high_channels), ("low_channels", low_channels),
            ("compressed_channels", compressed_channels), ("resample_groups", resample_groups),
            ("chunk_size", chunk_size),
        ):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if low_channels % resample_groups:
            raise ValueError("FreqFusion low_channels must be divisible by resample_groups.")
        if compressed_channels < 8:
            raise ValueError("FreqFusion compressed_channels must be at least eight.")
        for kernel in (lowpass_kernel, highpass_kernel):
            if type(kernel) is not int or not 3 <= kernel <= 15 or kernel % 2 != 1:
                raise ValueError("FreqFusion kernels must be odd integers from 3 to 15.")
        if type(checkpoint_reassembly) is not bool:
            raise ValueError("checkpoint_reassembly must be true or false.")

        self.high_channels, self.low_channels = high_channels, low_channels
        self.lowpass_kernel, self.highpass_kernel = lowpass_kernel, highpass_kernel
        self.checkpoint_reassembly, self.chunk_size = checkpoint_reassembly, chunk_size
        self.high_compress = nn.Conv2d(high_channels, compressed_channels, 1)
        self.low_compress = nn.Conv2d(low_channels, compressed_channels, 1)
        self.lowpass_generator = nn.Conv2d(compressed_channels, lowpass_kernel ** 2, 3, padding=1)
        self.highpass_generator = nn.Conv2d(compressed_channels, highpass_kernel ** 2, 3, padding=1)
        for layer in (self.high_compress, self.low_compress):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        for layer in (self.lowpass_generator, self.highpass_generator):
            nn.init.normal_(layer.weight, std=0.001)
            nn.init.zeros_(layer.bias)
        for name, kernel in (("lowpass_window", lowpass_kernel), ("highpass_window", highpass_kernel)):
            window = torch.hamming_window(kernel, periodic=False)
            self.register_buffer(name, torch.outer(window, window).reshape(1, kernel ** 2, 1, 1))
        self.sampler = _SimilaritySampler(compressed_channels, resample_groups)

    @staticmethod
    def _normalize_kernel(logits: torch.Tensor, window: torch.Tensor) -> torch.Tensor:
        # The normalization is stable if an AMP-enabled engine is added later.
        weights = logits.float().softmax(dim=1) * window.float()
        return (weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)).to(logits.dtype)

    def _filter(
        self, features: torch.Tensor, weights: torch.Tensor, kernel: int, scale: int
    ) -> torch.Tensor:
        target_height, target_width = weights.shape[-2:]
        full_height, full_width = (size * scale for size in features.shape[-2:])
        if target_height > full_height or target_width > full_width:
            raise ValueError("Dynamic filter kernel map exceeds its supported upsampling size.")
        if target_height != full_height or target_width != full_width:
            weights = F.pad(
                weights, (0, full_width - target_width, 0, full_height - target_height), mode="replicate"
            )
        outputs = []
        for chunk in features.split(self.chunk_size, dim=1):
            if self.checkpoint_reassembly and torch.is_grad_enabled() and (
                chunk.requires_grad or weights.requires_grad
            ):
                filtered = checkpoint(
                    _reassemble_chunk, chunk, weights, kernel, scale, use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                filtered = _reassemble_chunk(chunk, weights, kernel, scale)
            outputs.append(filtered[:, :, :target_height, :target_width])
        return torch.cat(outputs, dim=1)

    def forward(
        self, high: torch.Tensor, low: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if high.ndim != 4 or low.ndim != 4:
            raise ValueError("FreqFusion requires two B,C,H,W feature maps.")
        if high.shape[0] != low.shape[0] or high.shape[1] != self.high_channels or low.shape[1] != self.low_channels:
            raise ValueError("FreqFusion inputs have incompatible batch/channel dimensions.")
        if any(size < 1 for size in high.shape[-2:] + low.shape[-2:]):
            raise ValueError("FreqFusion feature maps must have positive spatial dimensions.")
        if any(high_size not in (2 * low_size - 1, 2 * low_size)
               for high_size, low_size in zip(high.shape[-2:], low.shape[-2:])):
            raise ValueError("FreqFusion expects adjacent MiT sizes: high=2*low or 2*low-1.")
        high_compressed = self.high_compress(high)
        low_compressed = self.low_compress(low)

        # Enhanced initial fusion: use high-resolution detail to route coarse
        # generator logits before jointly generating the final spatial kernels.
        high_logits = self.highpass_generator(high_compressed)
        initial_high_weights = self._normalize_kernel(high_logits, self.highpass_window)
        high_compressed = 2.0 * high_compressed - self._filter(
            high_compressed, initial_high_weights, self.highpass_kernel, 1
        )
        low_logits = self.lowpass_generator(high_compressed)
        initial_low_weights = self._normalize_kernel(low_logits, self.lowpass_window)
        low_logits = low_logits + self._filter(
            self.lowpass_generator(low_compressed), initial_low_weights, self.lowpass_kernel, 2
        )
        low_weights = self._normalize_kernel(low_logits, self.lowpass_window)
        high_logits = high_logits + self._filter(
            self.highpass_generator(low_compressed), low_weights, self.lowpass_kernel, 2
        )
        high_weights = self._normalize_kernel(high_logits, self.highpass_window)

        enhanced_high = 2.0 * high - self._filter(high, high_weights, self.highpass_kernel, 1)
        aligned_low = self._filter(low, low_weights, self.lowpass_kernel, 2)
        aligned_low = self.sampler(high_compressed, low_compressed, aligned_low)
        return enhanced_high, aligned_low
