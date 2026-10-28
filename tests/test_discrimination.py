"""Behavioral checks for optional encoder feedback; no weights/data download.

Run with: python -m unittest discover -s tests -v
The small MiT used here preserves the real four-stage encoder contract.
"""

from __future__ import annotations

import copy
import tempfile
import unittest
from unittest import mock

import torch
from torch import nn
from torch.nn import functional as F
from transformers import SegformerConfig, SegformerForImageClassification, SegformerForSemanticSegmentation

from segformer_baseline.config import default_config, validate_config
from segformer_baseline.discrimination import DiscriminativeEncoder, ForegroundBackgroundDiscrimination
from segformer_baseline.engine import resized_logits, training_losses
from segformer_baseline.model import (
    _attach_architecture,
    _new_model,
    architecture_from_config,
    build_model,
    model_from_checkpoint,
    model_parameter_counts,
    optimizer_parameter_groups,
)


def tiny_config() -> SegformerConfig:
    return SegformerConfig(
        num_channels=3,
        num_labels=2,
        hidden_sizes=[8, 16, 20, 32],
        depths=[1, 1, 1, 1],
        num_attention_heads=[1, 2, 4, 4],
        sr_ratios=[4, 2, 2, 1],
        decoder_hidden_size=16,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        classifier_dropout_prob=0.0,
        drop_path_rate=0.0,
        reshape_last_stage=True,
    )


def experiment_config(**overrides) -> dict:
    config = default_config()
    config.update(
        pretrained=False,
        discrimination_channels=8,
        freqfusion_compressed_channels=8,
        freqfusion_resample_groups=2,
        freqfusion_lowpass_kernel=3,
        freqfusion_highpass_kernel=3,
        freqfusion_chunk_size=4,
        upsample_channels=[8, 4],
    )
    config.update(overrides)
    return config


def tiny_model(**overrides):
    config = experiment_config(**overrides)
    architecture = architecture_from_config(config)
    torch.manual_seed(42)
    model = _new_model(tiny_config(), architecture)
    _attach_architecture(model, architecture)
    return model, config


def payload_for(model, config) -> dict:
    return {
        "model": copy.deepcopy(model.state_dict()),
        "model_config": copy.deepcopy(model.config.to_dict()),
        "architecture": copy.deepcopy(model.config.local_refinement),
        "config": copy.deepcopy(config),
        "normalization_mean": [0.0, 0.0, 0.0],
        "normalization_std": [1.0, 1.0, 1.0],
    }


class BranchBehaviorTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(3)

    def test_zero_initialized_feedback_preserves_features_exactly(self):
        branch = ForegroundBackgroundDiscrimination(16, width=8).eval()
        features = torch.randn(2, 16, 7, 9)
        refined, coarse = branch(features)
        torch.testing.assert_close(refined, features, rtol=0, atol=0)
        self.assertEqual(coarse.shape, (2, 2, 7, 9))

    def test_references_do_not_mix_different_images(self):
        branch = ForegroundBackgroundDiscrimination(16, width=8).eval()
        nn.init.normal_(branch.residual_projection.weight, std=0.2)
        image = torch.randn(1, 16, 7, 9)
        unrelated = torch.randn_like(image) * 15 + 20
        alone, alone_coarse = branch(image)
        together, together_coarse = branch(torch.cat((image, unrelated)))
        torch.testing.assert_close(alone, together[:1], rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(alone_coarse, together_coarse[:1], rtol=1e-5, atol=1e-6)

    def test_auxiliary_mode_has_no_feature_feedback(self):
        branch = ForegroundBackgroundDiscrimination(16, width=8, mode="auxiliary")
        features = torch.randn(2, 16, 7, 9)
        refined, coarse = branch(features)
        self.assertIs(refined, features)
        self.assertFalse(hasattr(branch, "residual_projection"))
        loss = F.cross_entropy(coarse, torch.ones((2, 7, 9), dtype=torch.long))
        loss.backward()
        self.assertGreater(branch.coarse_classifier.weight.grad.abs().sum().item(), 0)

    def test_negligible_class_mass_and_zero_features_stay_finite(self):
        branch = ForegroundBackgroundDiscrimination(16, width=8)
        with torch.no_grad():
            branch.coarse_classifier.weight.zero_()
            branch.coarse_classifier.bias.copy_(torch.tensor([1000.0, -1000.0]))
            branch.residual_projection.weight.normal_(std=0.1)
        refined, coarse = branch(torch.zeros(2, 16, 3, 5))
        self.assertTrue(torch.isfinite(refined).all())
        self.assertTrue(torch.isfinite(coarse).all())
        refined.square().mean().backward()
        for parameter in branch.parameters():
            if parameter.grad is not None:
                self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_cpu_autocast_with_extreme_coarse_logits_has_finite_backward(self):
        branch = ForegroundBackgroundDiscrimination(16, width=8)
        with torch.no_grad():
            branch.coarse_classifier.weight.zero_()
            branch.coarse_classifier.bias.copy_(torch.tensor([1000.0, -1000.0]))
            branch.residual_projection.weight.normal_(std=0.1)
        features = torch.randn(2, 16, 7, 9, requires_grad=True)
        target = torch.ones((2, 7, 9), dtype=torch.long)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            refined, coarse = branch(features)
            loss = refined.float().square().mean() + 0.2 * F.cross_entropy(coarse.float(), target)
        self.assertEqual(coarse.dtype, torch.bfloat16)
        self.assertTrue(torch.isfinite(refined).all())
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(features.grad).all())
        self.assertGreater(features.grad.abs().sum().item(), 0)
        for parameter in branch.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertGreater(branch.residual_projection.weight.grad.abs().sum().item(), 0)
        self.assertGreater(branch.coarse_classifier.bias.grad.abs().sum().item(), 0)


class EncoderIntegrationTests(unittest.TestCase):
    def test_local_pretrained_encoder_keeps_all_weights_and_remaps_input_bands(self):
        torch.manual_seed(17)
        source = SegformerForImageClassification(tiny_config())
        first_layer_key = "segformer.encoder.patch_embeddings.0.proj.weight"
        with torch.no_grad():
            # Unequal RGB filters make an incorrect channel permutation observable.
            source.segformer.encoder.patch_embeddings[0].proj.weight[:, 0].fill_(1.0)
            source.segformer.encoder.patch_embeddings[0].proj.weight[:, 1].fill_(3.0)
            source.segformer.encoder.patch_embeddings[0].proj.weight[:, 2].fill_(8.0)
        expected_source = copy.deepcopy(source.state_dict())
        with tempfile.TemporaryDirectory(prefix="segformer-local-pretrained-") as directory:
            source.save_pretrained(directory)
            config = experiment_config(
                pretrained=True,
                pretrained_model=directory,
                encoder_discrimination=True,
            )
            with mock.patch("requests.sessions.Session.request", side_effect=AssertionError("network call")):
                model = build_model(config)
        actual = model.state_dict()
        copied_encoder_keys = [key for key in expected_source if key.startswith("segformer.")]
        self.assertGreater(len(copied_encoder_keys), 0)
        for key in copied_encoder_keys:
            with self.subTest(tensor=key):
                expected = expected_source[key]
                if key == first_layer_key:
                    expected = torch.stack((expected.mean(dim=1), expected[:, 0], expected[:, 1]), dim=1)
                torch.testing.assert_close(actual[key], expected, rtol=0, atol=0)
        branch_ids = {id(parameter) for parameter in model.segformer.encoder.discrimination.parameters()}
        groups = optimizer_parameter_groups(model, config)
        self.assertTrue(branch_ids)
        self.assertTrue(branch_ids.issubset({id(parameter) for parameter in groups[1]["params"]}))
        self.assertEqual(groups[1]["lr"], config["decoder_lr"])

    def test_disabled_switch_preserves_huggingface_state_and_output(self):
        torch.manual_seed(42)
        reference = SegformerForSemanticSegmentation(tiny_config()).eval()
        model, _ = tiny_model()
        model.eval()
        self.assertEqual(list(reference.state_dict()), list(model.state_dict()))
        for key, tensor in reference.state_dict().items():
            torch.testing.assert_close(tensor, model.state_dict()[key], rtol=0, atol=0)
        image = torch.randn(1, 3, 65, 67)
        with torch.inference_mode():
            torch.testing.assert_close(reference(image).logits, model(image).logits, rtol=0, atol=0)

    def test_feedback_reaches_next_encoder_stage_and_decoder_feature(self):
        base = SegformerForSemanticSegmentation(tiny_config()).segformer.encoder.eval()
        reference = copy.deepcopy(base)
        architecture = architecture_from_config(experiment_config(encoder_discrimination=True))
        encoder = DiscriminativeEncoder(base, tiny_config(), architecture).eval()
        image = torch.randn(1, 3, 65, 67)
        with torch.inference_mode():
            original = reference(image, output_hidden_states=True, return_dict=True)
            initial = encoder(image, output_hidden_states=True, return_dict=True)
        for expected, actual in zip(original.hidden_states, initial.hidden_states):
            torch.testing.assert_close(expected, actual, rtol=0, atol=0)

        nn.init.normal_(encoder.discrimination.residual_projection.weight, std=0.2)
        stage_three_inputs = []
        hook = encoder.patch_embeddings[2].register_forward_pre_hook(
            lambda module, args: stage_three_inputs.append(args[0].detach().clone())
        )
        try:
            with torch.inference_mode():
                changed = encoder(image, output_hidden_states=True, return_dict=True)
        finally:
            hook.remove()
        torch.testing.assert_close(changed.hidden_states[0], original.hidden_states[0], rtol=0, atol=0)
        self.assertFalse(torch.allclose(changed.hidden_states[1], original.hidden_states[1]))
        torch.testing.assert_close(stage_three_inputs[0], changed.hidden_states[1], rtol=0, atol=0)
        self.assertFalse(torch.allclose(changed.hidden_states[2], original.hidden_states[2]))
        self.assertFalse(torch.allclose(changed.hidden_states[3], original.hidden_states[3]))

    def test_auxiliary_control_keeps_all_encoder_features_unchanged(self):
        model, _ = tiny_model(encoder_discrimination=True, discrimination_mode="auxiliary")
        reference, _ = tiny_model()
        image = torch.randn(1, 3, 64, 64)
        model.eval()
        reference.eval()
        with torch.inference_mode():
            actual = model(image, output_hidden_states=True)
            expected = reference(image, output_hidden_states=True)
        torch.testing.assert_close(actual.logits, expected.logits, rtol=0, atol=0)
        for left, right in zip(actual.hidden_states, expected.hidden_states):
            torch.testing.assert_close(left, right, rtol=0, atol=0)

    def test_optimizer_covers_every_parameter_once_at_expected_rates(self):
        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                model, config = tiny_model(encoder_discrimination=enabled)
                groups = optimizer_parameter_groups(model, config)
                identifiers = [id(parameter) for group in groups for parameter in group["params"]]
                self.assertEqual(len(identifiers), len(set(identifiers)))
                self.assertEqual(set(identifiers), {id(parameter) for parameter in model.parameters()})
                self.assertEqual([group["lr"] for group in groups], [config["encoder_lr"], config["decoder_lr"]])
                if enabled:
                    branch_ids = {id(parameter) for parameter in model.segformer.encoder.discrimination.parameters()}
                    self.assertTrue(branch_ids.issubset({id(parameter) for parameter in groups[1]["params"]}))
                    self.assertTrue(branch_ids.isdisjoint({id(parameter) for parameter in groups[0]["params"]}))
                    counts = model_parameter_counts(model)
                    self.assertGreater(counts["encoder_discrimination"], 0)
                    self.assertEqual(counts["total"], counts["encoder_backbone"] + counts["encoder_discrimination"] + counts["decoder"])
                else:
                    self.assertEqual([id(p) for p in groups[0]["params"]], [id(p) for p in model.segformer.parameters()])
                    self.assertEqual([id(p) for p in groups[1]["params"]], [id(p) for p in model.decode_head.parameters()])

    def test_foreground_and_background_training_have_finite_loss_and_gradients(self):
        for label in (0, 1):
            with self.subTest(label=label):
                model, _ = tiny_model(encoder_discrimination=True)
                model.eval()  # Avoid testing unrelated batch-statistics behavior.
                image = torch.randn(2, 3, 64, 64)
                target = torch.full((2, 64, 64), label, dtype=torch.long)
                losses = training_losses(model, image, target, nn.CrossEntropyLoss())
                expected = losses["main_loss"] + losses["auxiliary_weight"] * losses["auxiliary_loss"]
                torch.testing.assert_close(losses["loss"], expected)
                self.assertTrue(torch.isfinite(losses["loss"]))
                losses["loss"].backward()
                branch = model.segformer.encoder.discrimination
                self.assertGreater(branch.coarse_classifier.weight.grad.abs().sum().item(), 0)
                self.assertGreater(branch.residual_projection.weight.grad.abs().sum().item(), 0)
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_labels_api_matches_engine_training_objective(self):
        model, _ = tiny_model(encoder_discrimination=True)
        model.eval()
        image = torch.randn(1, 3, 65, 67)
        target = torch.randint(0, 2, (1, 65, 67))
        with torch.inference_mode():
            output = model(image, labels=target)
            losses = training_losses(model, image, target, nn.CrossEntropyLoss())
        torch.testing.assert_close(output.loss, losses["loss"])

    def test_old_checkpoint_without_discrimination_fields_loads_offline(self):
        model, config = tiny_model()
        payload = payload_for(model, config)
        new_fields = [key for key in config if key.startswith("discrimination_") or key == "encoder_discrimination"]
        for description in (payload["architecture"], payload["config"], payload["model_config"]["local_refinement"]):
            for key in new_fields:
                description.pop(key, None)
        with mock.patch("transformers.SegformerForImageClassification.from_pretrained", side_effect=AssertionError("network call")):
            restored = model_from_checkpoint(payload).eval()
        self.assertFalse(restored.config.local_refinement["encoder_discrimination"])
        model.eval()
        image = torch.randn(1, 3, 64, 64)
        with torch.inference_mode():
            torch.testing.assert_close(restored(image).logits, model(image).logits, rtol=0, atol=0)

    def test_checkpoint_roundtrip_for_original_and_custom_decoders(self):
        variants = (
            {},
            {"decoder_fusion": "progressive", "decoder_upsampling": "learned"},
            {"freqfusion": True},
            {"decoder_fusion": "progressive", "decoder_upsampling": "learned", "freqfusion": True},
        )
        for variant in variants:
            with self.subTest(variant=variant):
                model, config = tiny_model(encoder_discrimination=True, **variant)
                nn.init.normal_(model.segformer.encoder.discrimination.residual_projection.weight, std=0.1)
                model.eval()
                payload = payload_for(model, config)
                with mock.patch("transformers.SegformerForImageClassification.from_pretrained", side_effect=AssertionError("network call")):
                    restored = model_from_checkpoint(payload).eval()
                image = torch.randn(1, 3, 65, 67)
                with torch.inference_mode():
                    before = resized_logits(model, image, (65, 67))
                    after = resized_logits(restored, image, (65, 67))
                self.assertEqual(after.shape, (1, 2, 65, 67))
                torch.testing.assert_close(before, after, rtol=0, atol=0)
                self.assertEqual(list(model.state_dict()), list(restored.state_dict()))

    def test_checkpoint_conflicting_architecture_is_rejected(self):
        model, config = tiny_model(encoder_discrimination=True)
        payload = payload_for(model, config)
        payload["architecture"]["encoder_discrimination"] = False
        with self.assertRaisesRegex(ValueError, "metadata disagree"):
            model_from_checkpoint(payload)

    def test_invalid_configuration_is_rejected(self):
        invalid = (
            {"encoder_discrimination": "true"},
            {"discrimination_mode": "other"},
            {"discrimination_channels": 1},
            {"discrimination_temperature": 0},
            {"discrimination_temperature": float("nan")},
            {"discrimination_aux_weight": -0.1},
            {"encoder_discrimination": True, "discrimination_mode": "auxiliary", "discrimination_aux_weight": 0},
        )
        for overrides in invalid:
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                validate_config(experiment_config(**overrides))


if __name__ == "__main__":
    unittest.main()
