"""Checks for training-only mining from existing foreground/background masks."""

import unittest

import torch

from segformer_baseline.config import default_config, validate_config
from segformer_baseline.contrastive import HardAlgaeContrastiveLoss


class HardAlgaeContrastiveTests(unittest.TestCase):
    def setUp(self):
        config = default_config()
        config.update(contrastive_samples=1, input_channels=[1, 2, 3])
        self.criterion = HardAlgaeContrastiveLoss(config)
        self.images = torch.zeros(1, 3, 32, 32)
        self.masks = torch.zeros(1, 32, 32, dtype=torch.long)
        self.masks[0, 16, 8:16] = 1
        self.images[0, 2, 16, 8:16] = torch.arange(1, 9)
        self.logits = torch.zeros(1, 2, 32, 32)
        self.logits[0, 1, 16, 8:10] = torch.tensor([-1, -2])
        self.logits[0, 1, 16, 14:16] = torch.tensor([1, 2])
        self.logits[0, 1, 3, 3] = 3

    def test_selections_use_nir_rank_and_model_margin_within_each_label(self):
        weak, clear, background = self.criterion._select(self.images, self.masks, self.logits)
        self.assertEqual([weak.item(), clear.item(), background.item()], [16 * 32 + 9, 16 * 32 + 15, 3 * 32 + 3])

    def test_loss_prefers_clear_algae_and_backpropagates_through_features(self):
        features = torch.zeros(1, 2, 32, 32, requires_grad=True)
        with torch.no_grad():
            features[0, 0, 16, 9] = 1
            features[0, 0, 16, 15] = 1
            features[0, 1, 3, 3] = 1
        better = self.criterion(features, self.images, self.masks, self.logits)
        swapped = features.detach().clone()
        swapped[0, :, 16, 15] = features[0, :, 3, 3].detach()
        swapped[0, :, 3, 3] = features[0, :, 16, 15].detach()
        worse = self.criterion(swapped, self.images, self.masks, self.logits)
        self.assertLess(better.item(), worse.item())
        better.backward()
        self.assertTrue(torch.isfinite(features.grad).all())
        self.assertGreater(features.grad.abs().sum().item(), 0)

    def test_batches_without_separable_labeled_groups_skip_contrast(self):
        features = torch.zeros(1, 2, 32, 32)
        self.assertIsNone(self.criterion(features, self.images, torch.zeros_like(self.masks), self.logits))
        self.assertIsNone(self.criterion(features, self.images, torch.ones_like(self.masks), self.logits))
        self.assertIsNone(self.criterion(features, torch.zeros_like(self.images), self.masks, self.logits))

    def test_training_config_requires_learned_decoder_and_nir(self):
        config = default_config()
        config["loss"] = "cross_entropy_contrastive"
        with self.assertRaisesRegex(ValueError, "learned upsampling"):
            validate_config(config)
        config.update(decoder_upsampling="learned", input_channels=[2, 1, 0])
        with self.assertRaisesRegex(ValueError, "NIR"):
            validate_config(config)
        config["input_channels"] = [3, 2, 1]
        validate_config(config)


if __name__ == "__main__":
    unittest.main()
