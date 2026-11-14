"""Input-scale controls keep loss/metrics on native labels and survive resume."""

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
from segformer_baseline.input_scale import (
    checkpoint_input_scale, forward_with_input_scale, model_input_scale, scaled_size,
)
from segformer_baseline.lovasz import foreground_lovasz_loss
from segformer_baseline.model import (
    _attach_architecture, _new_model, architecture_from_config, checkpoint_model_signature,
    model_from_checkpoint, model_parameter_counts,
)
from test_unravel import tiny_config


class RecordingModel(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.config = SimpleNamespace(input_scale=scale)
        self.conv = nn.Conv2d(3, 2, 1)
        self.seen = []

    def forward(self, pixel_values):
        self.seen.append(pixel_values)
        return SimpleNamespace(logits=self.conv(pixel_values))


def settings(scale=2.0):
    result = default_config()
    result.update(input_scale=scale, pretrained=False, loss="cross_entropy_lovasz",
                  decoder_fusion="progressive", decoder_upsampling="learned", freqfusion=True,
                  upsample_channels=[8,4], freqfusion_compressed_channels=8, freqfusion_resample_groups=2,
                  freqfusion_chunk_size=4, freqfusion_lowpass_kernel=3, freqfusion_highpass_kernel=3)
    return result


def tiny_model(scale):
    torch.manual_seed(42)
    config = settings(scale)
    model_config = tiny_config()
    model_config.input_scale = scale
    architecture = architecture_from_config(config)
    model = _new_model(model_config, architecture)
    _attach_architecture(model, architecture)
    return model, config


def payload(model, config):
    return {"input_scale": config["input_scale"], "config": copy.deepcopy(config),
            "model_config": model.config.to_dict(), "architecture": model.config.local_refinement,
            "model": copy.deepcopy(model.state_dict())}


class InputScaleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_scale_one_is_identity_and_default_models_keep_the_legacy_path(self):
        image = torch.randn(2,3,65,67)
        model = RecordingModel(1.0)
        expected = model(pixel_values=image).logits
        actual = resized_logits(model, image, image.shape[-2:])
        self.assertIs(model.seen[-1], image)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        del model.config.input_scale
        self.assertEqual(model_input_scale(model), 1.0)
        forward_with_input_scale(model, image)
        self.assertIs(model.seen[-1], image)

    def test_images_are_scaled_but_ce_lovasz_and_gradients_use_unchanged_original_masks(self):
        model = RecordingModel(2.0)
        image = torch.randn(2,3,35,37)
        masks = torch.randint(0,2,(2,35,37))
        original = masks.clone()
        result = training_losses(model,image,masks,nn.CrossEntropyLoss(),lovasz_weight=.1)
        self.assertEqual(model.seen[-1].shape[-2:], (70,74))
        expected_input = F.interpolate(image, (70,74), mode="bilinear", align_corners=False, antialias=False)
        torch.testing.assert_close(model.seen[-1], expected_input, rtol=0, atol=0)
        expected_logits = F.interpolate(model.conv(expected_input), (35,37), mode="bilinear", align_corners=False)
        expected_loss = F.cross_entropy(expected_logits,masks) + .1*foreground_lovasz_loss(expected_logits,masks)
        torch.testing.assert_close(result["loss"], expected_loss, rtol=0, atol=0)
        torch.testing.assert_close(masks, original, rtol=0, atol=0)
        result["loss"].backward()
        self.assertGreater(model.conv.weight.grad.abs().sum(), 0)
        self.assertTrue(torch.isfinite(model.conv.weight.grad).all())

    def test_evaluation_counts_original_pixels_and_exports_original_size_predictions(self):
        model = RecordingModel(2.0)
        image = torch.randn(2,3,33,35)
        masks = torch.randint(0,2,(2,33,35))
        loader=[dict(image=image,mask=masks,name=["a.tif","b.tif"],image_path=["a.tif","b.tif"])]
        result=evaluate_model(model,loader,torch.device("cpu"),nn.CrossEntropyLoss(),sample_limit=2)
        self.assertEqual(result["confusion"].sum(), masks.numel())
        self.assertEqual(result["num_images"], 2)
        self.assertEqual(len(result["per_image"]), 2)
        self.assertEqual(result["samples"][0]["prediction"].shape, (33,35))
        self.assertEqual(model.seen[-1].shape[-2:], (66,70))

    def test_input_scale_adds_no_parameters_or_rng_changes_and_restores_offline(self):
        unscaled, _ = tiny_model(1.0)
        expected_rng = torch.get_rng_state().clone()
        scaled, config = tiny_model(2.0)
        self.assertTrue(torch.equal(expected_rng,torch.get_rng_state()))
        self.assertEqual(model_parameter_counts(unscaled),model_parameter_counts(scaled))
        for key,value in unscaled.state_dict().items():
            torch.testing.assert_close(value,scaled.state_dict()[key],rtol=0,atol=0)
        scaled.eval()
        checkpoint=payload(scaled,config)
        restored=model_from_checkpoint(checkpoint).eval()
        self.assertEqual(model_input_scale(restored),2.0)
        image=torch.randn(2,3,65,67)
        with torch.inference_mode():
            torch.testing.assert_close(resized_logits(scaled,image,(65,67)),
                                       resized_logits(restored,image,(65,67)),rtol=0,atol=0)

    def test_legacy_checkpoint_defaults_and_signatures_normalize_scale_one(self):
        model,config=tiny_model(1.0)
        current=payload(model,config)
        legacy=copy.deepcopy(current)
        for container in (legacy,legacy["config"],legacy["model_config"]):
            container.pop("input_scale",None)
        self.assertEqual(checkpoint_input_scale(legacy),1.0)
        self.assertEqual(checkpoint_model_signature(legacy),checkpoint_model_signature(current))
        restored=model_from_checkpoint(legacy)
        self.assertEqual(model_input_scale(restored),1.0)

    def test_contradictory_metadata_and_resume_scale_changes_are_rejected(self):
        model,config=tiny_model(2.0)
        checkpoint=payload(model,config)
        for container_key in ("config","model_config"):
            bad=copy.deepcopy(checkpoint)
            bad[container_key]["input_scale"]=1.0
            with self.assertRaisesRegex(ValueError,"metadata disagree"):
                model_from_checkpoint(bad)
        checkpoint.update(schema_version=2,optimizer={},scheduler={},rng_state={},history=[],
                          dataset_fingerprint="synthetic",best_iou=.5,best_epoch=1)
        args=argparse.Namespace(config=None,resume=Path("run/checkpoints/last.pt"),input_scale=1.0)
        with self.assertRaisesRegex(ValueError,"Cannot change input_scale"):
            train.prepare_config(args,checkpoint)
        args.input_scale=None
        self.assertEqual(train.prepare_config(args,checkpoint)["input_scale"],2.0)

    def test_config_is_a_pure_scale_control_and_dimensions_round_half_up(self):
        root=Path(__file__).resolve().parents[1]
        base=dict(default_config(),**read_config(root/"configs/segformer_b0_decoder_freqfusion_lovasz.yaml"))
        scaled=dict(default_config(),**read_config(root/"configs/segformer_b0_decoder_freqfusion_lovasz_scale2.yaml"))
        validate_config(scaled)
        self.assertEqual({key for key in base if base[key]!=scaled[key]},{"input_scale"})
        self.assertEqual(scaled_size((65,67),.5),(33,34))
        self.assertEqual(scaled_size((256,256),2.0),(512,512))
        for value in (True,0,-1,5,float("nan"),float("inf"),"2"):
            with self.subTest(value=value),self.assertRaises(ValueError):
                validate_config(dict(base,input_scale=value))
        with self.assertRaisesRegex(ValueError,"too small"):
            forward_with_input_scale(RecordingModel(.5),torch.randn(1,3,32,32))


if __name__=="__main__":
    unittest.main()
