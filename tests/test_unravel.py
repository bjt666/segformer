"""Behavioral tests for the F2-only adapter and offline checkpoint handling."""

from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from torch import nn
from transformers import SegformerConfig, SegformerForImageClassification, SegformerForSemanticSegmentation

from segformer_baseline.config import default_config, read_config, validate_config
from segformer_baseline.engine import resized_logits, training_losses
from segformer_baseline.model import (
    _attach_architecture, _new_model, architecture_from_config, build_model,
    model_from_checkpoint, model_parameter_counts, optimizer_parameter_groups,
)
from segformer_baseline.unravel import F2UnravelAdapter, UnravelEncoder, _GaussianFilter


def settings(**updates):
    config = default_config()
    config.update(pretrained=False, encoder_unravel=True, unravel_channels=32,
                  unravel_quality_channels=16, unravel_kernel=3, upsample_channels=[8, 4],
                  freqfusion_compressed_channels=8, freqfusion_resample_groups=2,
                  freqfusion_lowpass_kernel=3, freqfusion_highpass_kernel=3, freqfusion_chunk_size=4)
    config.update(updates)
    return config


def tiny_config():
    return SegformerConfig(num_channels=3, num_labels=2, hidden_sizes=[8, 16, 20, 32],
                           depths=[1, 1, 1, 1], num_attention_heads=[1, 2, 4, 4],
                           sr_ratios=[4, 2, 2, 1], decoder_hidden_size=16,
                           hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0,
                           classifier_dropout_prob=0.0, drop_path_rate=0.0)


def tiny_model(**updates):
    config = settings(**updates)
    architecture = architecture_from_config(config)
    torch.manual_seed(42)
    model = _new_model(tiny_config(), architecture)
    _attach_architecture(model, architecture)
    return model, config


def payload(model, config):
    return {"model": copy.deepcopy(model.state_dict()), "model_config": model.config.to_dict(),
            "architecture": copy.deepcopy(model.config.local_refinement), "config": copy.deepcopy(config)}


class F2UnravelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_shapes_constants_and_gradients_include_all_four_paths(self):
        gaussian = _GaussianFilter(4)
        torch.testing.assert_close(gaussian(torch.ones(1, 4, 5, 7)), torch.ones(1, 4, 5, 7))
        for shape in ((2, 16, 7, 9), (1, 16, 1, 1)):
            with self.subTest(shape=shape):
                adapter = F2UnravelAdapter(16, width=32, quality_width=16, kernel=3)
                image = torch.randn(shape, requires_grad=True)
                output = adapter(image)
                self.assertEqual(output.shape, image.shape)
                self.assertTrue(torch.isfinite(output).all())
                output.square().mean().backward()
                self.assertTrue(torch.isfinite(image.grad).all())
                for parameter in adapter.parameters():
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(torch.isfinite(parameter.grad).all())
                if shape[0] > 1:
                    for name in ("point_gate", "local", "directional", "quality"):
                        self.assertGreater(sum(p.grad.abs().sum().item() for p in getattr(adapter, name).parameters()), 0)

    def test_disabled_switch_keeps_original_weights_rng_and_predictions(self):
        torch.manual_seed(42)
        reference = SegformerForSemanticSegmentation(tiny_config()).eval()
        expected_rng = torch.get_rng_state().clone()
        model, _ = tiny_model(encoder_unravel=False)
        self.assertTrue(torch.equal(expected_rng, torch.get_rng_state()))
        self.assertEqual(list(reference.state_dict()), list(model.state_dict()))
        model.eval()
        image = torch.randn(1, 3, 65, 67)
        with torch.inference_mode():
            torch.testing.assert_close(reference(image).logits, model(image).logits, rtol=0, atol=0)

    def test_f2_is_used_by_both_stage_three_and_decoder_and_f1_is_unchanged(self):
        model, _ = tiny_model()
        reference, _ = tiny_model(encoder_unravel=False)
        image = torch.randn(1, 3, 65, 67)
        model.eval()
        reference.eval()
        stage_three_inputs = []
        hook = model.segformer.encoder.patch_embeddings[2].register_forward_pre_hook(
            lambda module, args: stage_three_inputs.append(args[0].detach().clone()))
        try:
            with torch.inference_mode():
                before = reference(image, output_hidden_states=True)
                after = model(image, output_hidden_states=True)
        finally:
            hook.remove()
        torch.testing.assert_close(before.hidden_states[0], after.hidden_states[0], rtol=0, atol=0)
        torch.testing.assert_close(stage_three_inputs[0], after.hidden_states[1], rtol=0, atol=0)
        for index in (1, 2, 3):
            self.assertGreater((before.hidden_states[index] - after.hidden_states[index]).abs().max().item(), 0)
        self.assertGreater((before.logits - after.logits).abs().max().item(), 0)
        with torch.no_grad():
            model.segformer.encoder.unravel.residual_scale.zero_()
        with torch.inference_mode():
            disabled = model(image, output_hidden_states=True)
        for expected, actual in zip(before.hidden_states, disabled.hidden_states):
            torch.testing.assert_close(expected, actual, rtol=0, atol=0)

    def test_local_pretraining_preserves_mit_and_remaps_nir_rg_without_network(self):
        source = SegformerForImageClassification(tiny_config())
        key = "segformer.encoder.patch_embeddings.0.proj.weight"
        with torch.no_grad():
            source.segformer.encoder.patch_embeddings[0].proj.weight[:, 0].fill_(1)
            source.segformer.encoder.patch_embeddings[0].proj.weight[:, 1].fill_(3)
            source.segformer.encoder.patch_embeddings[0].proj.weight[:, 2].fill_(8)
        original = copy.deepcopy(source.state_dict())
        with tempfile.TemporaryDirectory(prefix="unravel-pretrained-") as directory:
            source.save_pretrained(directory)
            with mock.patch("requests.sessions.Session.request", side_effect=AssertionError("network call")):
                model = build_model(settings(pretrained=True, pretrained_model=directory))
        for name, value in original.items():
            if name.startswith("segformer."):
                if name == key:
                    value = torch.stack((value.mean(dim=1), value[:, 0], value[:, 1]), dim=1)
                torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)

    def test_count_and_optimizer_partition_include_adapter_at_decoder_lr(self):
        model, config = tiny_model()
        groups = optimizer_parameter_groups(model, config)
        ids = [id(p) for group in groups for p in group["params"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(ids), {id(p) for p in model.parameters()})
        adapter_ids = {id(p) for p in model.segformer.encoder.unravel.parameters()}
        self.assertTrue(adapter_ids.issubset({id(p) for p in groups[1]["params"]}))
        self.assertTrue(adapter_ids.isdisjoint({id(p) for p in groups[0]["params"]}))
        self.assertEqual([g["lr"] for g in groups], [config["encoder_lr"], config["decoder_lr"]])
        counts = model_parameter_counts(model)
        self.assertEqual(counts["total"], counts["mit_encoder"] + counts["encoder_unravel"] + counts["decoder"])
        self.assertGreater(counts["encoder_unravel"], 0)

    def test_offline_roundtrip_and_training_for_original_and_combined_decoders(self):
        for combined in (False, True):
            with self.subTest(combined=combined):
                variant = dict(decoder_fusion="progressive", decoder_upsampling="learned", freqfusion=True) if combined else {}
                model, config = tiny_model(**variant)
                model.eval()
                image = torch.randn(2, 3, 65, 67)
                target = torch.randint(0, 2, (2, 65, 67))
                losses = training_losses(model, image, target, nn.CrossEntropyLoss(), lovasz_weight=0.1)
                self.assertIsNotNone(losses["lovasz_loss"])
                losses["loss"].backward()
                self.assertGreater(model.segformer.encoder.unravel.output_projection.weight.grad.abs().sum().item(), 0)
                optimizer = torch.optim.AdamW(optimizer_parameter_groups(model, config))
                optimizer.step()
                checkpoint = payload(model, config)
                checkpoint["optimizer"] = optimizer.state_dict()
                with mock.patch("transformers.SegformerForImageClassification.from_pretrained", side_effect=AssertionError("network call")):
                    restored = model_from_checkpoint(checkpoint).eval()
                restored_optimizer = torch.optim.AdamW(optimizer_parameter_groups(restored, config))
                restored_optimizer.load_state_dict(copy.deepcopy(checkpoint["optimizer"]))
                with torch.inference_mode():
                    torch.testing.assert_close(resized_logits(model, image, (65, 67)),
                                               resized_logits(restored, image, (65, 67)), rtol=0, atol=0)
                for current, current_optimizer in ((model, optimizer), (restored, restored_optimizer)):
                    current_optimizer.zero_grad()
                    training_losses(current, image, target, nn.CrossEntropyLoss(), lovasz_weight=0.1)["loss"].backward()
                    current_optimizer.step()
                for name, value in model.state_dict().items():
                    torch.testing.assert_close(value, restored.state_dict()[name], rtol=0, atol=0)

    def test_old_checkpoint_and_new_conflicting_metadata(self):
        model, config = tiny_model(encoder_unravel=False)
        old = payload(model, config)
        keys = [key for key in config if key.startswith("unravel_") or key == "encoder_unravel"]
        for description in (old["architecture"], old["config"], old["model_config"]["local_refinement"]):
            for key in keys:
                description.pop(key, None)
        restored = model_from_checkpoint(old)
        self.assertFalse(isinstance(restored.segformer.encoder, UnravelEncoder))
        self.assertEqual(list(model.state_dict()), list(restored.state_dict()))
        model, config = tiny_model()
        bad = payload(model, config)
        bad["architecture"]["encoder_unravel"] = False
        with self.assertRaisesRegex(ValueError, "metadata disagree"):
            model_from_checkpoint(bad)

    def test_all_new_configs_and_invalid_settings(self):
        root = Path(__file__).resolve().parents[1]
        for name in ("segformer_b0_lovasz", "segformer_b0_unravel_f2",
                     "segformer_b0_decoder_freqfusion_lovasz", "segformer_b0_decoder_freqfusion_unravel_f2"):
            config = default_config()
            config.update(read_config(root / "configs" / f"{name}.yaml"))
            validate_config(config)
        for invalid in ({"encoder_unravel": "true"}, {"unravel_channels": 17},
                        {"unravel_quality_channels": 15}, {"unravel_kernel": 4},
                        {"unravel_residual_scale": 0}, {"unravel_residual_scale": float("nan")},
                        {"dual_encoder": True}, {"encoder_discrimination": True}, {"module_a": True}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                validate_config(settings(**invalid))


if __name__ == "__main__":
    unittest.main()
