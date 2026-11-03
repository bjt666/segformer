"""Hybrid behavior, pretrained compatibility and offline checkpoint tests."""
from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from torch import nn
from transformers import SegformerForImageClassification

from segformer_baseline.config import validate_config
from segformer_baseline.engine import training_losses
from segformer_baseline.hybrid import DualEncoder, ResNet50Encoder
from segformer_baseline.model import (
    _attach_architecture, _new_model, architecture_from_config, build_model,
    model_from_checkpoint, model_parameter_counts, optimizer_parameter_groups,
)
from test_discrimination import tiny_config, experiment_config


def hybrid(interaction=True, highres=True):
    config = experiment_config(dual_encoder=True, dual_interaction=interaction, dual_highres=highres,
                               dual_highres_channels=8, dual_highres_blocks=1,
                               decoder_fusion="progressive", decoder_upsampling="learned")
    architecture = architecture_from_config(config)
    torch.manual_seed(42)
    model = _new_model(tiny_config(), architecture)
    _attach_architecture(model, architecture)
    return model, config


class HybridTests(unittest.TestCase):
    def test_resnet_trunk_matches_reference_implementation(self):
        try:
            from torchvision.models import resnet50
        except ImportError:
            self.skipTest("Optional torchvision reference is not installed")
        reference = resnet50(weights=None).eval()
        encoder = ResNet50Encoder().eval()
        state = {key: value for key, value in reference.state_dict().items() if not key.startswith("fc.")}
        encoder.load_state_dict(state, strict=True)
        self.assertEqual(sum(parameter.numel() for parameter in encoder.parameters()), 23508032)
        image = torch.randn(1, 3, 65, 67)
        with torch.inference_mode():
            expected = reference.relu(reference.bn1(reference.conv1(image)))
            actual = encoder.stem(image)
            torch.testing.assert_close(expected, actual, rtol=0, atol=0)
            expected, actual = reference.maxpool(expected), encoder.maxpool(actual)
            for stage in range(1, 5):
                expected, actual = getattr(reference, f"layer{stage}")(expected), getattr(encoder, f"layer{stage}")(actual)
                torch.testing.assert_close(expected, actual, rtol=0, atol=0)

    def test_same_stage_exchange_feeds_both_next_stages_and_decoder(self):
        model, _ = hybrid()
        model.eval()
        encoder = model.segformer.encoder
        captures = {}
        handles = [
            encoder.exchanges[0].register_forward_hook(lambda module, args, out: captures.update(exchange=out)),
            encoder.patch_embeddings[1].register_forward_pre_hook(lambda module, args: captures.update(mit_input=args[0])),
            encoder.cnn.layer2.register_forward_pre_hook(lambda module, args: captures.update(cnn_input=args[0])),
            model.decode_head.learned_upsampling.high_resolution_fusion.register_forward_pre_hook(
                lambda module, args: captures.update(half_inputs=args)),
        ]
        try:
            with torch.inference_mode():
                output = model(torch.randn(1, 3, 65, 67), output_hidden_states=True, output_attentions=True)
        finally:
            for handle in handles:
                handle.remove()
        self.assertEqual(output.logits.shape, (1, 2, 65, 67))
        self.assertEqual(len(output.hidden_states), 4)
        self.assertEqual(len(output.attentions), 4)
        self.assertIs(captures["mit_input"], captures["exchange"][0])
        self.assertIs(captures["cnn_input"], captures["exchange"][1])
        self.assertIs(output.hidden_states[0], captures["exchange"][2])
        self.assertEqual(captures["half_inputs"][0].shape[-2:], (33, 34))
        self.assertEqual(captures["half_inputs"][1].shape[-2:], (33, 34))

    def test_all_stages_and_highres_receive_main_loss_gradients(self):
        model, _ = hybrid()
        model.train()
        image = torch.randn(2, 3, 64, 64)
        target = torch.randint(0, 2, (2, 64, 64))
        losses = training_losses(model, image, target, nn.CrossEntropyLoss())
        self.assertEqual(losses["auxiliary_weight"], 0)
        self.assertEqual(losses["auxiliary_loss"].item(), 0)
        losses["loss"].backward()
        representatives = [model.segformer.encoder.cnn.conv1.weight,
                           model.segformer.encoder.high_resolution_branch[0][0].weight,
                           model.decode_head.learned_upsampling.high_resolution_fusion.gate.weight]
        for exchange in model.segformer.encoder.exchanges:
            representatives.extend((exchange.to_mit.weight, exchange.to_cnn.weight, exchange.decoder_projection[0].weight))
        for parameter in representatives:
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0)

    def test_optimizer_partition_and_parameter_accounting(self):
        model, config = hybrid()
        groups = optimizer_parameter_groups(model, config)
        ids = [id(parameter) for group in groups for parameter in group["params"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(ids), {id(parameter) for parameter in model.parameters()})
        backbone_ids = {id(parameter) for parameter in groups[0]["params"]}
        self.assertTrue({id(parameter) for parameter in model.segformer.encoder.cnn.parameters()}.issubset(backbone_ids))
        self.assertTrue({id(parameter) for parameter in model.segformer.encoder.new_parameters()}.isdisjoint(backbone_ids))
        counts = model_parameter_counts(model)
        self.assertEqual(counts["cnn_encoder"], 23508032)
        self.assertEqual(counts["total"], counts["mit_encoder"] + counts["cnn_encoder"] + counts["cross_interaction"]
                         + counts["high_resolution_branch"] + counts["decoder"])

    def test_ablation_switches_and_serialized_offline_roundtrip(self):
        for interaction, highres in ((False, False), (True, False), (False, True), (True, True)):
            with self.subTest(interaction=interaction, highres=highres), tempfile.TemporaryDirectory() as directory:
                model, config = hybrid(interaction, highres)
                model.eval()
                self.assertIsInstance(model.segformer.encoder, DualEncoder)
                self.assertEqual(hasattr(model.segformer.encoder, "high_resolution_branch"), highres)
                self.assertEqual(hasattr(model.segformer.encoder.exchanges[0], "to_cnn"), interaction)
                image = torch.randn(1, 3, 65, 67)
                with torch.inference_mode():
                    before = model(image).logits
                payload = {"model": model.state_dict(), "model_config": model.config.to_dict(), "config": config,
                           "architecture": copy.deepcopy(model.config.local_refinement)}
                path = Path(directory) / "checkpoint.pt"
                torch.save(payload, path)
                del payload, model
                with mock.patch("transformers.SegformerForImageClassification.from_pretrained", side_effect=AssertionError("network")), \
                        mock.patch("segformer_baseline.hybrid.cnn_weights", side_effect=AssertionError("network")):
                    restored = model_from_checkpoint(torch.load(path, weights_only=False)).eval()
                with torch.inference_mode():
                    after = restored(image).logits
                torch.testing.assert_close(before, after, rtol=0, atol=0)
                del restored

    def test_both_local_pretrained_encoders_and_band_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            source = SegformerForImageClassification(tiny_config())
            source.save_pretrained(Path(directory) / "mit")
            cnn = ResNet50Encoder()
            with torch.no_grad():
                cnn.conv1.weight[:, 0].fill_(1)
                cnn.conv1.weight[:, 1].fill_(3)
                cnn.conv1.weight[:, 2].fill_(8)
            cnn_path = Path(directory) / "cnn.pt"
            torch.save(cnn.state_dict(), cnn_path)
            config = experiment_config(pretrained=True, pretrained_model=str(Path(directory) / "mit"),
                                       cnn_pretrained_model=str(cnn_path), dual_encoder=True,
                                       decoder_fusion="progressive", decoder_upsampling="learned")
            with mock.patch("requests.sessions.Session.request", side_effect=AssertionError("network")):
                model = build_model(config)
            actual = model.segformer.encoder.cnn.state_dict()
            for key, value in cnn.state_dict().items():
                if key != "conv1.weight":
                    torch.testing.assert_close(actual[key], value, rtol=0, atol=0)
            torch.testing.assert_close(actual["conv1.weight"][:, 0], torch.full_like(actual["conv1.weight"][:, 0], 4))
            torch.testing.assert_close(actual["conv1.weight"][:, 1], torch.full_like(actual["conv1.weight"][:, 1], 1))
            torch.testing.assert_close(actual["conv1.weight"][:, 2], torch.full_like(actual["conv1.weight"][:, 2], 3))
            for key, value in source.segformer.state_dict().items():
                if key != "encoder.patch_embeddings.0.proj.weight":
                    torch.testing.assert_close(model.segformer.state_dict()[key], value, rtol=0, atol=0)

    def test_incompatible_combinations_are_rejected(self):
        for override in ({"encoder_discrimination": True}, {"module_a": True}, {"decoder_upsampling": "bilinear"},
                         {"dual_highres_blocks": 0}, {"dual_residual_scale": 0}):
            config = experiment_config(dual_encoder=True, decoder_fusion="progressive", decoder_upsampling="learned")
            config.update(override)
            with self.subTest(override=override), self.assertRaises(ValueError):
                validate_config(config)


if __name__ == "__main__":
    unittest.main()
