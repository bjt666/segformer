"""Checks for foreground Lovász supervision and pure-background fallback."""

import unittest
from types import SimpleNamespace

import torch

from segformer_baseline.engine import training_losses
from segformer_baseline.lovasz import foreground_lovasz_loss


class ForegroundLovaszTests(unittest.TestCase):
    def setUp(self):
        foreground_logits = torch.tensor([0.2, 0.9, 0.6]).logit().reshape(3, 1, 1)
        self.logits = torch.stack((torch.zeros_like(foreground_logits), foreground_logits), dim=1)
        self.logits.requires_grad_()
        self.masks = torch.tensor([1, 0, 1]).reshape(3, 1, 1)

    def test_batch_ranking_penalizes_foreground_misses_and_background_false_positives(self):
        loss = foreground_lovasz_loss(self.logits, self.masks)
        # Ranked errors are 0.9, 0.8, 0.4 with Jaccard increments of one third.
        torch.testing.assert_close(loss, torch.tensor(0.7))
        loss.backward()
        self.assertTrue(torch.isfinite(self.logits.grad).all())
        self.assertEqual(self.logits.grad[:, 1, 0, 0].sign().tolist(), [-1.0, 1.0, -1.0])

    def test_training_combines_terms_and_keeps_ce_on_background_only_batches(self):
        def model(*, pixel_values):
            return SimpleNamespace(logits=self.logits)

        ce = torch.nn.CrossEntropyLoss()
        mixed = training_losses(model, self.logits, self.masks, ce, lovasz_weight=0.1)
        torch.testing.assert_close(mixed["loss"], mixed["main_loss"] + 0.07)
        background = training_losses(model, self.logits, torch.zeros_like(self.masks), ce, lovasz_weight=0.1)
        self.assertIsNone(background["lovasz_loss"])
        torch.testing.assert_close(background["loss"], background["main_loss"])
