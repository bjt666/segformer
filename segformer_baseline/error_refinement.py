"""Supervised FP/FN risk prediction and bounded foreground-logit correction.

This is an experimental adaptation, not a reproduction of PUGNet. The forward
path never reads labels. Labels only construct detached error targets for the
training loss or measure errors during evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .decoder import DecoderFeatureOutput, DecoderSegformerForSemanticSegmentation


def _resize(values: torch.Tensor, size) -> torch.Tensor:
    return values if values.shape[-2:] == tuple(size) else F.interpolate(
        values, size=size, mode="bilinear", align_corners=False)


def _block(in_channels: int, width: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, width, 1, bias=False), nn.GroupNorm(4, width), nn.GELU(),
        nn.Conv2d(width, width, 3, padding=1, groups=width, bias=False),
        nn.Conv2d(width, width, 1, bias=False), nn.GroupNorm(4, width), nn.GELU(),
    )


class ErrorRefinement(nn.Module):
    """Use F1/F4 context and detached coarse probabilities at input resolution.

    Risk channel 0 estimates FP risk among coarse foreground pixels; channel 1
    estimates FN risk among coarse background pixels. The gated mode lowers
    foreground logits only in the first domain, and raises them only in the
    second. Corrections are bounded by ``max_correction`` in logit units.
    Class-balanced risk supervision does not produce calibrated probabilities.
    """

    def __init__(self, hidden_sizes, width: int, mode: str, max_correction: float):
        super().__init__()
        self.mode = mode
        self.max_correction = float(max_correction)
        self.shallow = nn.Conv2d(hidden_sizes[0], width, 1)
        self.deep = nn.Conv2d(hidden_sizes[-1], width, 1)
        self.context = _block(2 * width + 1, width)
        self.risk_head = nn.Conv2d(width, 2, 1)
        if mode != "auxiliary":
            self.correction_head = nn.Sequential(_block(width, width), nn.Conv2d(width, 2, 1))
            # Begin with very small changes while allowing gradients to reach
            # both heads immediately; no zero-initialized dead amplitude gate.
            nn.init.normal_(self.correction_head[-1].weight, std=0.001)
            nn.init.constant_(self.correction_head[-1].bias, -4.0)

    def forward(self, hidden_states, coarse_logits):
        size = coarse_logits.shape[-2:]
        foreground = coarse_logits.detach().softmax(dim=1)[:, 1:2]
        context = self.context(torch.cat((
            _resize(self.shallow(hidden_states[0]), size),
            _resize(self.deep(hidden_states[-1]), size), foreground,
        ), dim=1))
        risk_logits = self.risk_head(context)
        if self.mode == "auxiliary":
            correction = torch.zeros_like(coarse_logits[:, :1])
        else:
            amplitude = self.correction_head(context).sigmoid() * self.max_correction
            if self.mode == "plain":
                # Same trainable layers/parameter count as gated mode, but
                # neither risk maps nor prediction domains guide correction.
                correction = amplitude[:, 1:2] - amplitude[:, :1]
            else:
                domains = prediction_domains(coarse_logits)
                risk = risk_logits.sigmoid() * domains.to(risk_logits.dtype)
                correction = risk[:, 1:2] * amplitude[:, 1:2] - risk[:, :1] * amplitude[:, :1]
        logits = torch.cat((coarse_logits[:, :1], coarse_logits[:, 1:2] + correction), dim=1)
        return logits, risk_logits, correction


@torch.no_grad()
def prediction_domains(coarse_logits: torch.Tensor) -> torch.Tensor:
    foreground = coarse_logits.detach().argmax(dim=1, keepdim=True) == 1
    return torch.cat((foreground, ~foreground), dim=1)


@torch.no_grad()
def error_targets(coarse_logits: torch.Tensor, masks: torch.Tensor):
    """Return FP/FN labels and their separate domains; ignore nonbinary labels."""
    coarse_logits = _resize(coarse_logits.detach(), masks.shape[-2:])
    domains = prediction_domains(coarse_logits)
    valid = ((masks == 0) | (masks == 1)).unsqueeze(1)
    domains = domains & valid
    targets = torch.cat(((masks == 0).unsqueeze(1), (masks == 1).unsqueeze(1)), dim=1) & domains
    return targets, domains


def balanced_error_loss(risk_logits: torch.Tensor, coarse_logits: torch.Tensor,
                        masks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Average positive/negative BCE within each head, then average active heads.

    Missing classes/domains contribute no fictitious samples. This prevents
    the abundant correct pixels from overwhelming the few actual errors.
    """
    targets, domains = error_targets(coarse_logits, masks)
    risk_logits = _resize(risk_logits, masks.shape[-2:])
    elementwise = F.binary_cross_entropy_with_logits(risk_logits, targets.to(risk_logits.dtype), reduction="none")
    positive, negative = targets, domains & ~targets
    dimensions = (0, 2, 3)
    positive_count, negative_count = positive.sum(dimensions), negative.sum(dimensions)
    positive_loss = (elementwise * positive).sum(dimensions) / positive_count.clamp_min(1)
    negative_loss = (elementwise * negative).sum(dimensions) / negative_count.clamp_min(1)
    classes = (positive_count > 0).to(risk_logits.dtype) + (negative_count > 0).to(risk_logits.dtype)
    per_head = (positive_loss + negative_loss) / classes.clamp_min(1)
    return per_head.sum() / (classes > 0).sum().clamp_min(1), positive_count


@dataclass
class ErrorRefinementOutput(DecoderFeatureOutput):
    coarse_logits: torch.Tensor | None = None
    error_logits: torch.Tensor | None = None
    logit_correction: torch.Tensor | None = None


class ErrorRefinementSegformer(DecoderSegformerForSemanticSegmentation):
    """Add the correction after either the original or the redesigned decoder."""

    def forward(self, pixel_values, labels=None, output_attentions=None,
                output_hidden_states=None, return_dict=None, *, return_decoder_features=False):
        return_dict = self.config.use_return_dict if return_dict is None else return_dict
        expose_hidden = self.config.output_hidden_states if output_hidden_states is None else output_hidden_states
        outputs = super().forward(
            pixel_values, output_attentions=output_attentions, output_hidden_states=True,
            return_dict=True, return_decoder_features=return_decoder_features,
        )
        coarse_logits = _resize(outputs.logits, pixel_values.shape[-2:])
        logits, risk_logits, correction = self.error_refinement(outputs.hidden_states, coarse_logits)
        loss = None
        if labels is not None:
            settings = self.config.local_refinement
            final_ce = F.cross_entropy(_resize(logits, labels.shape[-2:]), labels,
                                       ignore_index=self.config.semantic_loss_ignore_index)
            coarse_ce = F.cross_entropy(_resize(coarse_logits, labels.shape[-2:]), labels,
                                        ignore_index=self.config.semantic_loss_ignore_index)
            risk_loss, _ = balanced_error_loss(risk_logits, coarse_logits, labels)
            weight = settings["error_coarse_weight"]
            # HF-style label loss is CE-based; the training engine adds the
            # selected Lovasz term separately, as for the existing models.
            loss = (1 - weight) * final_ce + weight * coarse_ce + settings["error_risk_weight"] * risk_loss
        result = ErrorRefinementOutput(
            loss=loss, logits=logits, hidden_states=outputs.hidden_states if expose_hidden else None,
            attentions=outputs.attentions, decoder_features=getattr(outputs, "decoder_features", None),
            coarse_logits=coarse_logits, error_logits=risk_logits, logit_correction=correction,
        )
        return result if return_dict else result.to_tuple()
