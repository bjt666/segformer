"""Behavior, supervision, ablation and offline recovery of error refinement."""

import copy
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from torch import nn

from segformer_baseline.config import default_config, read_config, validate_config
from segformer_baseline.engine import training_losses
from segformer_baseline.error_diagnostics import ErrorDiagnostics
from segformer_baseline.error_refinement import ErrorRefinement, balanced_error_loss, error_targets
from segformer_baseline.model import (
    _attach_architecture, _new_model, architecture_from_config, model_from_checkpoint,
    model_parameter_counts, optimizer_parameter_groups,
)
from test_unravel import tiny_config


def settings(**updates):
    config = default_config()
    config.update(pretrained=False, error_refinement=True, error_refinement_channels=8,
                  decoder_fusion="progressive", decoder_upsampling="learned", freqfusion=True,
                  upsample_channels=[8, 4], freqfusion_compressed_channels=8,
                  freqfusion_resample_groups=2, freqfusion_chunk_size=4,
                  freqfusion_lowpass_kernel=3, freqfusion_highpass_kernel=3,
                  loss="cross_entropy_lovasz")
    config.update(updates)
    return config


def tiny_model(**updates):
    config = settings(**updates)
    architecture = architecture_from_config(config)
    torch.manual_seed(42)
    model = _new_model(tiny_config(), architecture)
    _attach_architecture(model, architecture)
    return model, config


class ErrorRefinementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_error_labels_cover_fp_fn_correct_and_ignore_with_detached_targets(self):
        coarse = torch.tensor([[[[0., 0., 0., 0., 0.]], [[2., 2., -2., -2., 2.]]]], requires_grad=True)
        masks = torch.tensor([[[0, 1, 1, 0, 255]]])
        targets, domains = error_targets(coarse, masks)
        self.assertEqual(targets[0].tolist(), [[[True, False, False, False, False]],
                                            [[False, False, True, False, False]]])
        self.assertEqual(domains.sum().item(), 4)
        risk = torch.zeros_like(coarse, requires_grad=True)
        loss, count = balanced_error_loss(risk, coarse, masks)
        torch.testing.assert_close(loss, torch.tensor(np.log(2), dtype=loss.dtype))
        self.assertEqual(count.tolist(), [1, 1])
        loss.backward()
        self.assertIsNone(coarse.grad)
        self.assertLess(risk.grad[0, 0, 0, 0], 0)
        self.assertGreater(risk.grad[0, 0, 0, 1], 0)
        self.assertEqual(risk.grad[0, :, 0, 4].abs().sum(), 0)

    def test_balancing_absent_errors_empty_domains_and_all_ignored(self):
        coarse = torch.tensor([[[[0., 0.]], [[-2., -2.]]]])
        risk = torch.zeros_like(coarse, requires_grad=True)
        loss, count = balanced_error_loss(risk, coarse, torch.zeros(1, 1, 2, dtype=torch.long))
        torch.testing.assert_close(loss, torch.tensor(np.log(2), dtype=loss.dtype))
        self.assertEqual(count.tolist(), [0, 0])
        loss.backward()
        self.assertEqual(risk.grad[:, 0].abs().sum(), 0)
        self.assertGreater(risk.grad[:, 1].sum(), 0)
        ignored, _ = balanced_error_loss(risk, coarse, torch.full((1, 1, 2), 255))
        self.assertEqual(ignored.item(), 0)
        self.assertTrue(torch.isfinite(ignored))

    def test_direction_bound_and_risk_gating_affect_inference_without_labels(self):
        module = ErrorRefinement([8, 16, 20, 32], 8, "gated", 4)
        features = [torch.randn(2, 8, 3, 4), None, None, torch.randn(2, 32, 1, 1)]
        coarse = torch.randn(2, 2, 11, 13)
        prediction = coarse.argmax(dim=1)
        refined, risk, correction = module(features, coarse)
        self.assertEqual(refined.shape, coarse.shape)
        self.assertEqual(risk.shape, coarse.shape)
        self.assertTrue((correction[:, 0][prediction == 1] <= 0).all())
        self.assertTrue((correction[:, 0][prediction == 0] >= 0).all())
        self.assertLessEqual(correction.abs().max().item(), 4)
        torch.testing.assert_close(refined[:, 0], coarse[:, 0], rtol=0, atol=0)
        with torch.no_grad():
            module.risk_head.weight.zero_()
            module.risk_head.bias.fill_(-30)
        torch.testing.assert_close(module(features, coarse)[0], coarse, rtol=0, atol=1e-7)

    def test_same_shared_weights_rng_and_auxiliary_output_when_disabled_or_auxiliary(self):
        for decoder in (False, True):
            variant = {} if decoder else dict(decoder_fusion="mlp", decoder_upsampling="bilinear", freqfusion=False)
            reference, _ = tiny_model(error_refinement=False, **variant)
            expected_rng = torch.get_rng_state().clone()
            auxiliary, _ = tiny_model(error_refinement_mode="auxiliary", **variant)
            self.assertTrue(torch.equal(expected_rng, torch.get_rng_state()))
            for key, value in reference.state_dict().items():
                torch.testing.assert_close(value, auxiliary.state_dict()[key], rtol=0, atol=0)
            image = torch.randn(2, 3, 65, 67)
            reference.eval()
            auxiliary.eval()
            with torch.inference_mode():
                expected = reference(image).logits
                expected = torch.nn.functional.interpolate(expected, (65, 67), mode="bilinear", align_corners=False)
                result = auxiliary(image)
                torch.testing.assert_close(expected, result.logits, rtol=0, atol=0)
                self.assertIsNone(result.hidden_states)
                exposed = auxiliary(image, output_hidden_states=True)
                self.assertEqual(len(exposed.hidden_states), 4)

    def test_full_loss_gradients_counts_optimizer_and_offline_roundtrip(self):
        model, config = tiny_model()
        model.eval()
        images = torch.randn(2, 3, 65, 67)
        masks = torch.randint(0, 2, (2, 65, 67))
        losses = training_losses(model, images, masks, nn.CrossEntropyLoss(), lovasz_weight=0.1)
        torch.testing.assert_close(losses["loss"], 0.8 * losses["main_loss"] + 0.2 * losses["error_coarse_loss"]
                                   + 0.1 * losses["error_risk_loss"] + 0.1 * losses["lovasz_loss"])
        losses["loss"].backward()
        for parameter in model.error_refinement.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0)
        counts = model_parameter_counts(model)
        self.assertEqual(counts["total"], counts["encoder"] + counts["decoder"] + counts["error_refinement"])
        groups = optimizer_parameter_groups(model, config)
        ids = [id(p) for group in groups for p in group["params"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(ids), {id(p) for p in model.parameters()})
        self.assertTrue({id(p) for p in model.error_refinement.parameters()}.issubset({id(p) for p in groups[1]["params"]}))
        optimizer = torch.optim.AdamW(groups)
        optimizer.step()
        checkpoint = {"model": copy.deepcopy(model.state_dict()), "model_config": model.config.to_dict(),
                      "architecture": model.config.local_refinement, "config": config}
        with mock.patch("requests.sessions.Session.request", side_effect=AssertionError("network call")):
            restored = model_from_checkpoint(checkpoint).eval()
        restored_optimizer = torch.optim.AdamW(optimizer_parameter_groups(restored, config))
        restored_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
        with torch.inference_mode():
            torch.testing.assert_close(model(images).logits, restored(images).logits, rtol=0, atol=0)
        for current, current_optimizer in ((model, optimizer), (restored, restored_optimizer)):
            current_optimizer.zero_grad()
            training_losses(current, images, masks, nn.CrossEntropyLoss(), lovasz_weight=0.1)["loss"].backward()
            current_optimizer.step()
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, restored.state_dict()[key], rtol=0, atol=0)

    def test_plain_has_same_capacity_but_does_not_read_risk_in_inference(self):
        gated, _ = tiny_model()
        plain, _ = tiny_model(error_refinement_mode="plain")
        self.assertEqual(model_parameter_counts(gated), model_parameter_counts(plain))
        for left, right in zip(gated.state_dict().values(), plain.state_dict().values()):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        plain.eval()
        images = torch.randn(2, 3, 64, 64)
        with torch.inference_mode():
            before = plain(images).logits
            plain.error_refinement.risk_head.bias.add_(10)
            torch.testing.assert_close(before, plain(images).logits, rtol=0, atol=0)

    def test_labels_only_change_loss_and_label_api_matches_ce_training(self):
        model, _ = tiny_model()
        model.eval()
        images = torch.randn(2, 3, 64, 64)
        zeros = torch.zeros(2, 64, 64, dtype=torch.long)
        ones = torch.ones_like(zeros)
        with torch.no_grad():
            no_labels, background, foreground = model(images), model(images, labels=zeros), model(images, labels=ones)
            torch.testing.assert_close(no_labels.logits, background.logits, rtol=0, atol=0)
            torch.testing.assert_close(no_labels.logits, foreground.logits, rtol=0, atol=0)
            expected = training_losses(model, images, zeros, nn.CrossEntropyLoss())["loss"]
            torch.testing.assert_close(expected, background.loss)
            self.assertTrue(torch.isfinite(foreground.loss))

    def test_diagnostics_count_improved_and_harmed_pixels_and_detectable_errors(self):
        masks = torch.tensor([[[0, 1, 1, 0]]])
        coarse = torch.tensor([[[[0., 0., 0., 0.]], [[2., -2., 2., -2.]]]])
        final = torch.tensor([[[[0., 0., 0., 0.]], [[-2., 2., -2., -2.]]]])
        risk = torch.tensor([[[[8., -8., -8., -8.]], [[-8., 8., -8., -8.]]]])
        diagnostics = ErrorDiagnostics()
        diagnostics.update(coarse, final, risk, masks)
        report = diagnostics.report()
        self.assertEqual(report["initial_error_pixels"], 2)
        self.assertEqual(report["corrected_error_pixels"], 2)
        self.assertEqual(report["spoiled_correct_pixels"], 1)
        for head in report["risk_localization"].values():
            self.assertEqual(head["learned_risk"]["average_precision_histogram"], 1)
            self.assertEqual(head["coarse_entropy"]["average_precision_histogram"], 0.5)

    def test_old_metadata_and_invalid_settings(self):
        disabled, config = tiny_model(error_refinement=False)
        checkpoint = {"model": disabled.state_dict(), "model_config": disabled.config.to_dict(),
                      "config": copy.deepcopy(config)}
        for metadata in (checkpoint["config"], checkpoint["model_config"]["local_refinement"]):
            for key in list(metadata):
                if key.startswith("error_"):
                    metadata.pop(key)
        restored = model_from_checkpoint(checkpoint)
        self.assertEqual(list(restored.state_dict()), list(disabled.state_dict()))
        root = Path(__file__).resolve().parents[1]
        for suffix, mode in (("refinement", "gated"), ("auxiliary", "auxiliary"), ("plain", "plain")):
            config = default_config()
            config.update(read_config(root / "configs" / f"segformer_b0_decoder_freqfusion_error_{suffix}.yaml"))
            validate_config(config)
            self.assertTrue(config["error_refinement"])
            self.assertEqual(config["error_refinement_mode"], mode)
        for updates in ({"error_refinement": "true"}, {"error_refinement_mode": "unknown"},
                        {"error_refinement_channels": 7}, {"error_max_correction": float("nan")},
                        {"error_risk_weight": 0}, {"error_coarse_weight": 1}, {"error_coarse_weight": -1},
                        {"encoder_unravel": True}, {"encoder_discrimination": True}, {"dual_encoder": True},
                        {"module_a": True}, {"spectral_refinement": "local"}, {"loss": "cross_entropy_contrastive"}):
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                validate_config(settings(**updates))


if __name__ == "__main__":
    unittest.main()
