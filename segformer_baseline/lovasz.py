"""Foreground-only Lovász-Softmax for binary segmentation."""

import torch


def foreground_lovasz_loss(logits: torch.Tensor, masks: torch.Tensor) -> torch.Tensor | None:
    """Compute one Lovász term over all pixels in a batch with foreground."""
    foreground = masks.flatten() == 1
    if not foreground.any():
        return None

    # The sigmoid of the two-logit margin is the foreground softmax probability.
    probability = (logits[:, 1].float() - logits[:, 0].float()).sigmoid().flatten()
    errors = torch.where(foreground, 1 - probability, probability)
    sorted_errors, order = errors.sort(descending=True, stable=True)
    sorted_foreground = foreground[order].to(dtype=sorted_errors.dtype)

    positives = sorted_foreground.sum()
    intersection = positives - sorted_foreground.cumsum(dim=0)
    union = positives + (1 - sorted_foreground).cumsum(dim=0)
    jaccard_loss = 1 - intersection / union
    weights = jaccard_loss.diff(prepend=jaccard_loss.new_zeros(1))
    return sorted_errors.dot(weights)
