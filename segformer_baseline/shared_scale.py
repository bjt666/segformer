"""Shared-weight 1x/2x encoding with optional spatially gated feature residuals.

This is a supervised feature-fusion experiment inspired by multi-resolution
segmentation, not an HRDA reproduction or a claim of a new attention mechanism.
The native normalized image is supplied explicitly; it is never reconstructed
by downsampling the interpolated main input.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from transformers.modeling_outputs import SemanticSegmenterOutput

from .decoder import DecoderFeatureOutput, DecoderSegformerForSemanticSegmentation
from .input_scale import scaled_size


def _bottleneck(channels: int, width: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(channels, width, 1, bias=False), nn.GroupNorm(4, width), nn.GELU(),
        nn.Conv2d(width, width, 3, padding=1, groups=width, bias=False),
        nn.GroupNorm(4, width), nn.GELU(),
    )


class ScaleFeatureFusion(nn.Module):
    """Common residual candidate; interaction adds a learned per-position gate.

    Plain: H + delta([H,L]). Interaction: H + 2*sigmoid(g([H,L,|H-L|]))*delta.
    The gate starts at one, so matched plain/interaction initial functions are
    identical. Candidate weights are nonzero: context and both shared-encoder
    passes receive gradients from the first update. No GT/prediction masks or
    detached confidence rules choose a scale.
    """
    def __init__(self, channels: int, width: int, mode: str):
        super().__init__()
        self.candidate = nn.Sequential(_bottleneck(2 * channels, width), nn.Conv2d(width, channels, 1))
        nn.init.normal_(self.candidate[-1].weight, std=0.02)
        nn.init.zeros_(self.candidate[-1].bias)
        if mode == "interaction":
            # A separate RNG fork preserves the candidate initialization of the
            # next stage and the outer training RNG in the plain control.
            with torch.random.fork_rng(devices=[]):
                self.gate = nn.Sequential(_bottleneck(3 * channels, width), nn.Conv2d(width, 1, 1))
                nn.init.zeros_(self.gate[-1].weight)
                nn.init.zeros_(self.gate[-1].bias)
        elif mode != "plain":
            raise ValueError("ScaleFeatureFusion mode must be plain or interaction.")

    def forward(self, main: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        context = F.interpolate(context, size=main.shape[-2:], mode="bilinear", align_corners=False)
        delta = self.candidate(torch.cat((main, context), dim=1))
        if hasattr(self, "gate"):
            gate = 2 * self.gate(torch.cat((main, context, (main-context).abs()), dim=1)).sigmoid()
            delta = gate * delta
        return main + delta


class SharedScaleFusion(nn.Module):
    def __init__(self, hidden_sizes, architecture: dict):
        super().__init__()
        self.stages = tuple(sorted(architecture["shared_scale_stages"]))
        self.blocks = nn.ModuleDict({
            f"f{stage}": ScaleFeatureFusion(hidden_sizes[stage-1], architecture["shared_scale_width"],
                                            architecture["shared_scale"])
            for stage in self.stages
        })

    def forward(self, main, context):
        features = list(main)
        for stage in self.stages:
            features[stage-1] = self.blocks[f"f{stage}"](features[stage-1], context[stage-1])
        return tuple(features)


class SharedScaleSegformer(DecoderSegformerForSemanticSegmentation):
    """Two encoder calls share the SAME module; only one main-resolution decoder.

    Fusion is after the two complete encoder passes. It does not feed the fused
    F2/F3 back into later MiT stages. F1/F4 remain the main path's features.
    Returned hidden states are decoder inputs; attentions are the main path's.
    """
    def forward(self, pixel_values: torch.Tensor, labels=None, output_attentions=None,
                output_hidden_states=None, return_dict=None, *, context_pixel_values=None,
                return_decoder_features=False):
        if context_pixel_values is None:
            raise ValueError("SharedScaleSegformer requires native context_pixel_values; use forward_with_input_scale.")
        if (pixel_values.ndim != 4 or context_pixel_values.ndim != 4
                or pixel_values.shape[:2] != context_pixel_values.shape[:2]
                or min(context_pixel_values.shape[-2:]) < 32
                or tuple(pixel_values.shape[-2:]) != scaled_size(context_pixel_values.shape[-2:], 2.0)):
            raise ValueError("Shared-scale inputs must be matching BCHW tensors at native/2x sizes (native >=32).")
        output_hidden_states = (self.config.output_hidden_states if output_hidden_states is None
                                else output_hidden_states)
        output_attentions = self.config.output_attentions if output_attentions is None else output_attentions
        return_dict = self.config.use_return_dict if return_dict is None else return_dict
        main = self.segformer(pixel_values, output_attentions=output_attentions,
                              output_hidden_states=True, return_dict=True)
        context = self.segformer(context_pixel_values, output_attentions=False,
                                 output_hidden_states=True, return_dict=True)
        features = self.shared_scale_fusion(main.hidden_states, context.hidden_states)
        decoded = self.decode_head(features, input_size=pixel_values.shape[-2:], pixel_values=pixel_values,
                                   return_features=return_decoder_features)
        logits, decoder_features = decoded if return_decoder_features else (decoded, None)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(F.interpolate(logits, size=labels.shape[-2:], mode="bilinear",
                                                align_corners=False), labels,
                                   ignore_index=self.config.semantic_loss_ignore_index)
        if not return_dict:
            output = (logits,)
            if output_hidden_states:
                output += (features,)
            if main.attentions is not None:
                output += (main.attentions,)
            if return_decoder_features:
                output += (decoder_features,)
            return (loss,) + output if loss is not None else output
        fields = dict(loss=loss, logits=logits, hidden_states=features if output_hidden_states else None,
                      attentions=main.attentions)
        if return_decoder_features:
            return DecoderFeatureOutput(**fields, decoder_features=decoder_features)
        return SemanticSegmenterOutput(**fields)
