"""Training-only contrast between weak algae, clear algae and hard background."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .utils import box_sum


class HardAlgaeContrastiveLoss(nn.Module):
    """Rank weak foreground features against class-balanced cosine evidence."""

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.nir_index = config["input_channels"].index(3)
        self.temperature = config["contrastive_temperature"]
        self.sample_limit = config["contrastive_samples"]

    def _sample(self, candidates: torch.Tensor, margin: torch.Tensor, *, largest: bool) -> torch.Tensor:
        indices = margin[candidates].topk(min(self.sample_limit, candidates.numel()), largest=largest).indices
        return candidates[indices]

    @torch.no_grad()
    def _select(self, images: torch.Tensor, masks: torch.Tensor, logits: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        nir = images[:, self.nir_index:self.nir_index + 1].float()
        water = (masks == 0).unsqueeze(1)
        background = water.flatten().nonzero(as_tuple=True)[0]
        if not background.numel():
            return None
        neighborhood = box_sum(torch.cat((nir * water, water.float()), dim=1), 31)
        support = neighborhood[:, 1:2]
        eligible = ((masks == 1).unsqueeze(1) & (support >= 64)).flatten().nonzero(as_tuple=True)[0]
        quartile = eligible.numel() // 4
        if not quartile:
            return None
        contrast = (nir - neighborhood[:, :1] / support.clamp_min(1)).abs().flatten()[eligible]
        weak_values, weak_offsets = contrast.topk(quartile, largest=False)
        clear_values, clear_offsets = contrast.topk(quartile)
        # Equal contrasts do not provide evidence for a weak/clear distinction.
        weak = eligible[weak_offsets[weak_values < clear_values[-1] - 1e-6]]
        if not weak.numel():
            return None
        margin = (logits[:, 1].float() - logits[:, 0].float()).flatten()
        return (self._sample(weak, margin, largest=False),
                self._sample(eligible[clear_offsets], margin, largest=True),
                self._sample(background, margin, largest=True))

    def forward(self, features: torch.Tensor, images: torch.Tensor,
                masks: torch.Tensor, logits: torch.Tensor) -> torch.Tensor | None:
        selected = self._select(images, masks, logits)
        if selected is None:
            return None
        pixels = features.shape[-2] * features.shape[-1]
        spatial_features = features.flatten(2).transpose(1, 2)
        weak, clear, background = (
            F.normalize(spatial_features[ids // pixels, ids % pixels].float(), dim=1)
            for ids in selected
        )
        positive = (weak @ clear.T) / self.temperature
        negative = (weak @ background.T) / self.temperature
        # Average within each group so an uneven number of valid keys does not
        # change the relative weight of the positive and negative classes.
        positive = positive.logsumexp(dim=1) - math.log(clear.shape[0])
        negative = negative.logsumexp(dim=1) - math.log(background.shape[0])
        return F.softplus(negative - positive).mean()
