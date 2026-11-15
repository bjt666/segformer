"""Shared encoding, matched controls, native-grid losses and offline recovery."""
import argparse
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch
from torch import nn
from torch.nn import functional as F

import train
from segformer_baseline.config import default_config, read_config, validate_config
from segformer_baseline.engine import evaluate_model, resized_logits, training_losses
from segformer_baseline.input_scale import forward_with_input_scale, input_scale_policy
from segformer_baseline.model import (
    _attach_architecture, _new_model, architecture_from_config, model_from_checkpoint,
    model_parameter_counts, optimizer_parameter_groups,
)
from test_input_scale import settings, payload
from test_unravel import tiny_config


def make_model(mode):
    config = dict(settings(), shared_scale=mode, shared_scale_width=8)
    validate_config(config)
    mc = tiny_config()
    mc.input_scale = 2.0
    torch.manual_seed(42)
    architecture = architecture_from_config(config)
    model = _new_model(mc, architecture)
    _attach_architecture(model, architecture)
    return model, config


class SharedScaleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_native_context_is_original_tensor_and_main_is_exact_interpolation(self):
        class Recorder(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(input_scale=2.0, local_refinement={'shared_scale':'plain'})
            def forward(self, pixel_values, context_pixel_values):
                self.main, self.context = pixel_values, context_pixel_values
                return SimpleNamespace(logits=pixel_values[:, :2])
        model = Recorder()
        source = torch.randn(2, 3, 35, 37)
        forward_with_input_scale(model, source)
        self.assertIs(model.context, source)
        torch.testing.assert_close(model.main, F.interpolate(source, size=(70,74), mode='bilinear',
                                   align_corners=False, antialias=False), rtol=0, atol=0)

    def test_two_passes_share_parameters_and_both_receive_gradients(self):
        model, config = make_model('interaction')
        calls = []
        handle = model.segformer.register_forward_hook(lambda module, args, output: calls.append(id(module)))
        main = torch.randn(2,3,70,74,requires_grad=True)
        context = torch.randn(2,3,35,37,requires_grad=True)
        masks = torch.randint(0,2,(2,35,37))
        result = model(main, context_pixel_values=context, labels=masks)
        self.assertEqual(result.logits.shape, (2,2,70,74))
        result.loss.backward()
        self.assertEqual(calls, [id(model.segformer)]*2)
        self.assertGreater(main.grad.abs().sum(), 0)
        self.assertGreater(context.grad.abs().sum(), 0)
        self.assertTrue(torch.isfinite(context.grad).all())
        self.assertGreater(model.shared_scale_fusion.blocks['f2'].candidate[-1].weight.grad.abs().sum(), 0)
        self.assertGreater(model.shared_scale_fusion.blocks['f2'].gate[-1].weight.grad.abs().sum(), 0)
        groups = optimizer_parameter_groups(model, config)
        ids = [id(p) for group in groups for p in group['params']]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(ids), {id(p) for p in model.parameters()})
        self.assertEqual({id(p) for p in groups[0]['params']}, {id(p) for p in model.segformer.parameters()})
        self.assertTrue({id(p) for p in model.shared_scale_fusion.parameters()}.issubset(
            {id(p) for p in groups[1]['params']}))
        handle.remove()

    def test_plain_and_interaction_start_from_identical_functions_and_shared_weights(self):
        plain, _ = make_model('plain')
        rng = torch.get_rng_state().clone()
        interaction, _ = make_model('interaction')
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for name, value in plain.state_dict().items():
            torch.testing.assert_close(value, interaction.state_dict()[name], rtol=0, atol=0)
        image = torch.randn(2,3,35,37)
        with torch.no_grad():
            a = resized_logits(plain.eval(), image, (35,37))
            b = resized_logits(interaction.eval(), image, (35,37))
        torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_loss_metrics_and_hidden_states_use_expected_grids(self):
        model, _ = make_model('interaction')
        image = torch.randn(2,3,35,37)
        masks = torch.randint(0,2,(2,35,37))
        result = training_losses(model, image, masks, nn.CrossEntropyLoss(), lovasz_weight=.1)
        self.assertTrue(torch.isfinite(result['loss']))
        result['loss'].backward()
        loader = [dict(image=image, mask=masks, name=['a.tif','b.tif'], image_path=['a.tif','b.tif'])]
        evaluation = evaluate_model(model, loader, torch.device('cpu'), nn.CrossEntropyLoss(), sample_limit=2)
        self.assertEqual(evaluation['confusion'].sum(), masks.numel())
        self.assertEqual(evaluation['samples'][0]['prediction'].shape, (35,37))
        with torch.no_grad():
            output = forward_with_input_scale(model.eval(), image, output_hidden_states=True,
                                             output_attentions=True, return_decoder_features=True)
            self.assertEqual(len(output.hidden_states),4)
            self.assertEqual(output.hidden_states[0].shape[-2:], (18,19))
            self.assertEqual(output.decoder_features.shape[-2:], (70,74))
            self.assertEqual(len(output.attentions),4)
            output_tuple = forward_with_input_scale(model, image, return_dict=False)
            self.assertEqual(output_tuple[0].shape, (2,2,70,74))
        with self.assertRaisesRegex(ValueError, 'context_pixel_values'):
            model(image)

    def test_old_off_checkpoint_has_identical_state_and_no_new_modules(self):
        model, config = make_model('off')
        checkpoint = payload(model, config)
        for description in (checkpoint['architecture'], checkpoint['config'],
                            checkpoint['model_config']['local_refinement']):
            for key in ('shared_scale','shared_scale_stages','shared_scale_width'):
                description.pop(key, None)
        restored = model_from_checkpoint(checkpoint)
        self.assertFalse(hasattr(restored, 'shared_scale_fusion'))
        for name, tensor in model.state_dict().items():
            torch.testing.assert_close(tensor, restored.state_dict()[name], rtol=0, atol=0)
        self.assertEqual(model_parameter_counts(model)['shared_scale_fusion'], 0)

    def test_new_checkpoint_roundtrip_and_conflicting_architecture_rejected(self):
        model, config = make_model('interaction')
        checkpoint = payload(model, config)
        restored = model_from_checkpoint(checkpoint).eval()
        image = torch.randn(2,3,35,37)
        with torch.inference_mode():
            torch.testing.assert_close(resized_logits(model.eval(),image,(35,37)),
                                       resized_logits(restored,image,(35,37)),rtol=0,atol=0)
        corrupt = copy.deepcopy(checkpoint)
        corrupt['config']['shared_scale'] = 'plain'
        with self.assertRaisesRegex(ValueError, 'metadata disagree'):
            model_from_checkpoint(corrupt)
        checkpoint.update(schema_version=2,optimizer={},scheduler={},rng_state={},history=[],
                          dataset_fingerprint='synthetic',best_iou=.5,best_epoch=1)
        args = argparse.Namespace(config=None,resume=Path('run/checkpoints/last.pt'),shared_scale='plain')
        with self.assertRaisesRegex(ValueError,'Cannot change shared_scale'):
            train.prepare_config(args, checkpoint)

    def test_shipped_configs_change_only_the_mode_and_keep_the_strong_control(self):
        root = Path(__file__).resolve().parents[1]
        configs = {mode: dict(default_config(), **read_config(root/'configs'/name)) for mode, name in (
            ('off','segformer_b0_decoder_freqfusion_lovasz_scale2.yaml'),
            ('plain','segformer_b0_decoder_freqfusion_lovasz_shared_plain.yaml'),
            ('interaction','segformer_b0_decoder_freqfusion_lovasz_shared_interaction.yaml'))}
        for config in configs.values():
            validate_config(config)
        self.assertEqual({k for k in configs['off'] if configs['off'][k]!=configs['plain'][k]}, {'shared_scale'})
        self.assertEqual({k for k in configs['plain'] if configs['plain'][k]!=configs['interaction'][k]}, {'shared_scale'})
        self.assertIn('ONE shared MiT',input_scale_policy(configs['interaction']))

    def test_invalid_combinations_fail_before_training(self):
        _, config = make_model('plain')
        for updates in ({'input_scale':1.0},{'shared_scale':'bad'},{'shared_scale_stages':[1]},
                        {'shared_scale_stages':[2,2]},{'shared_scale_width':7},
                        {'error_refinement':True},{'encoder_unravel':True},{'dual_encoder':True},
                        {'decoder_upsampling':'bilinear'},{'spectral_refinement':'concat'}):
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                validate_config(dict(config,**updates))


if __name__ == '__main__':
    unittest.main()
