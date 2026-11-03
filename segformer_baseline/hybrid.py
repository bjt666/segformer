"""MiT/ResNet-50 feature exchange and a persistent half-resolution path.

ResNet bottlenecks use the standard 1x1/3x3/1x1 layout with stride in the
3x3 convolution (TorchVision ResNet v1.5 convention). State names match its
ImageNet weights; torchvision is not a runtime dependency. This hybrid is
an experimental adaptation, not a reproduction of HRNet or BiSeNet.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from transformers.modeling_outputs import BaseModelOutput, SemanticSegmenterOutput

from .decoder import DecoderSegformerForSemanticSegmentation


RESNET50_URL = "https://download.pytorch.org/models/resnet50-11ad3fa6.pth"


def cnn_weights(source: str) -> dict:
    """Read an explicit local file or the official, hash-checked torch cache."""
    if source != "imagenet1k_v2":
        path = Path(source).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"ResNet-50 weights not found: {path}")
        state = torch.load(path, map_location="cpu", weights_only=True)
    else:
        cached = Path(torch.hub.get_dir()) / "checkpoints" / RESNET50_URL.rsplit("/", 1)[-1]
        offline = any(os.environ.get(name, "").lower() in ("1", "true", "yes")
                      for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"))
        if offline and not cached.is_file():
            raise FileNotFoundError(
                "ResNet-50 is not cached and offline mode is enabled. Run download_pretrained.py "
                "--cnn-only --output-dir /root/models with networking, then pass "
                "--cnn-pretrained-model /root/models/resnet50_imagenet1k_v2.pth."
            )
        state = torch.hub.load_state_dict_from_url(RESNET50_URL, map_location="cpu", check_hash=True)
    if not isinstance(state, dict):
        raise ValueError("ResNet weights must be a state-dict mapping.")
    return state.get("state_dict", state)


class _Bottleneck(nn.Module):
    def __init__(self, incoming: int, width: int, stride: int = 1) -> None:
        super().__init__()
        outgoing = 4 * width
        self.conv1 = nn.Conv2d(incoming, width, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, outgoing, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(outgoing)
        self.relu = nn.ReLU(inplace=False)
        self.downsample = (nn.Sequential(nn.Conv2d(incoming, outgoing, 1, stride=stride, bias=False),
                                        nn.BatchNorm2d(outgoing))
                           if stride != 1 or incoming != outgoing else None)

    def forward(self, features):
        shortcut = features if self.downsample is None else self.downsample(features)
        features = self.relu(self.bn1(self.conv1(features)))
        features = self.relu(self.bn2(self.conv2(features)))
        return self.relu(self.bn3(self.conv3(features)) + shortcut)


class ResNet50Encoder(nn.Module):
    """Complete [3,4,6,3] ResNet-50 trunk; excludes average pooling and FC."""
    stage_channels = (256, 512, 1024, 2048)

    def __init__(self, input_channels: int = 3) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        incoming = 64
        for index, (width, depth) in enumerate(zip((64, 128, 256, 512), (3, 4, 6, 3)), start=1):
            blocks = [_Bottleneck(incoming, width, stride=1 if index == 1 else 2)]
            blocks += [_Bottleneck(4 * width, width) for _ in range(depth - 1)]
            setattr(self, f"layer{index}", nn.Sequential(*blocks))
            incoming = 4 * width
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def stem(self, images):
        return self.relu(self.bn1(self.conv1(images)))

    def load_pretrained(self, source: str, selected_channels: list[int]) -> None:
        state = dict(cnn_weights(source))
        state.pop("fc.weight", None)
        state.pop("fc.bias", None)
        filters = state.get("conv1.weight")
        if filters is None or filters.ndim != 4 or filters.shape[1] != 3:
            raise ValueError("ResNet-50 pretrained stem must contain RGB filters.")
        mapping = {0: filters[:, 2], 1: filters[:, 1], 2: filters[:, 0], 3: filters.mean(dim=1)}
        state["conv1.weight"] = torch.stack([mapping[channel] for channel in selected_channels], dim=1)
        self.load_state_dict(state, strict=True)
        logging.getLogger("segformer").info("Loaded ResNet-50 ImageNet trunk from %s", source)


def _project(incoming, outgoing, kernel=1):
    return nn.Sequential(nn.Conv2d(incoming, outgoing, kernel, padding=kernel // 2, bias=False),
                         nn.GroupNorm(1, outgoing), nn.GELU())


class CrossStageExchange(nn.Module):
    """Simultaneous residual messages, then multiscale decoder fusion.

    Both deltas depend on the original same-stage inputs, never on an already
    updated feature. All four CNN outputs have a consumer, including stage 4.
    The no-interaction control keeps decoder fusion but removes both feedback
    messages from the encoder pathways.
    """
    def __init__(self, mit_channels, cnn_channels, interaction=True, scale=0.1):
        super().__init__()
        self.interaction = interaction
        self.decoder_projection = _project(cnn_channels, mit_channels)
        self.fusion_scale = nn.Parameter(torch.full((1, mit_channels, 1, 1), scale))
        if interaction:
            self.cnn_projection = _project(cnn_channels, mit_channels)
            self.mit_projection = _project(mit_channels, mit_channels)
            self.context = _project(2 * mit_channels, mit_channels, kernel=3)
            self.gates = nn.Conv2d(mit_channels, 2 * mit_channels, 1)
            self.to_mit = nn.Conv2d(mit_channels, mit_channels, 1, bias=False)
            self.to_cnn = nn.Conv2d(mit_channels, cnn_channels, 1, bias=False)
            self.mit_scale = nn.Parameter(torch.full((1, mit_channels, 1, 1), scale))
            self.cnn_scale = nn.Parameter(torch.full((1, cnn_channels, 1, 1), scale))

    def forward(self, mit, cnn):
        if mit.shape[-2:] != cnn.shape[-2:]:
            raise ValueError("MiT and CNN stage dimensions must match before exchange.")
        if self.interaction:
            local, semantic = self.cnn_projection(cnn), self.mit_projection(mit)
            gate_m, gate_c = self.gates(self.context(torch.cat((semantic, local), dim=1))).sigmoid().chunk(2, dim=1)
            delta_m = self.to_mit(gate_m * local)
            delta_c = self.to_cnn(gate_c * semantic)
            mit = mit + self.mit_scale * delta_m
            cnn = cnn + self.cnn_scale * delta_c
        fused = mit + self.fusion_scale * self.decoder_projection(cnn)
        return mit, cnn, fused


class _SpatialResidual(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.blocks = nn.Sequential(_project(channels, channels, 3),
                                    nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                                    nn.GroupNorm(1, channels))

    def forward(self, features):
        return F.gelu(features + self.blocks(features))


class HighResolutionFusion(nn.Module):
    def __init__(self, semantic_channels, detail_channels, scale=0.1):
        super().__init__()
        self.detail_projection = _project(detail_channels, semantic_channels)
        self.gate = nn.Conv2d(2 * semantic_channels, semantic_channels, 1)
        self.refine = _project(2 * semantic_channels, semantic_channels, 3)
        self.scale = nn.Parameter(torch.full((1, semantic_channels, 1, 1), scale))

    def forward(self, semantic, detail):
        if detail.shape[-2:] != semantic.shape[-2:]:
            raise ValueError("High-resolution features must match the actual half-input dimensions.")
        detail = self.detail_projection(detail)
        gate = self.gate(torch.cat((semantic, detail), dim=1)).sigmoid()
        return semantic + self.scale * self.refine(torch.cat((semantic, gate * detail), dim=1))


@dataclass
class DualEncoderOutput(BaseModelOutput):
    high_resolution_features: torch.Tensor | None = None


class DualEncoder(nn.Module):
    def __init__(self, original_encoder, config, architecture):
        super().__init__()
        if len(config.hidden_sizes) != 4 or not config.reshape_last_stage:
            raise ValueError("Dual encoder requires four native MiT feature maps.")
        if tuple(config.strides) != (4, 2, 2, 2):
            raise ValueError("Dual encoder requires MiT strides [4,2,2,2] for ResNet alignment.")
        self.config = config
        self.patch_embeddings, self.block, self.layer_norm = (
            original_encoder.patch_embeddings, original_encoder.block, original_encoder.layer_norm)
        self.cnn = ResNet50Encoder(config.num_channels)
        self.exchanges = nn.ModuleList([
            CrossStageExchange(mit, cnn, architecture["dual_interaction"], architecture["dual_residual_scale"])
            for mit, cnn in zip(config.hidden_sizes, self.cnn.stage_channels)
        ])
        if architecture["dual_highres"]:
            width = architecture["dual_highres_channels"]
            self.high_resolution_branch = nn.Sequential(
                _project(64, width),
                *[_SpatialResidual(width) for _ in range(architecture["dual_highres_blocks"])],
            )

    def new_parameters(self):
        yield from self.exchanges.parameters()
        if hasattr(self, "high_resolution_branch"):
            yield from self.high_resolution_branch.parameters()

    def forward(self, pixel_values, output_attentions=False, output_hidden_states=False, return_dict=True):
        batch = pixel_values.shape[0]
        stem = self.cnn.stem(pixel_values)
        high_resolution = self.high_resolution_branch(stem) if hasattr(self, "high_resolution_branch") else None
        cnn, mit = self.cnn.maxpool(stem), pixel_values
        hidden, attentions = (() if output_hidden_states else None), (() if output_attentions else None)
        for index, (embedding, blocks, norm, exchange) in enumerate(
            zip(self.patch_embeddings, self.block, self.layer_norm, self.exchanges)
        ):
            mit, height, width = embedding(mit)
            for block in blocks:
                result = block(mit, height, width, output_attentions=output_attentions)
                mit = result[0]
                if output_attentions:
                    attentions += (result[1],)
            mit = norm(mit).reshape(batch, height, width, -1).permute(0, 3, 1, 2).contiguous()
            cnn = getattr(self.cnn, f"layer{index + 1}")(cnn)
            mit, cnn, fused = exchange(mit, cnn)
            if output_hidden_states:
                hidden += (fused,)
        if not return_dict:
            return tuple(value for value in (fused, hidden, attentions, high_resolution) if value is not None)
        return DualEncoderOutput(last_hidden_state=fused, hidden_states=hidden, attentions=attentions,
                                 high_resolution_features=high_resolution)


class DualSegformerForSemanticSegmentation(DecoderSegformerForSemanticSegmentation):
    def forward(self, pixel_values, labels=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        return_dict = self.config.use_return_dict if return_dict is None else return_dict
        output_hidden_states = self.config.output_hidden_states if output_hidden_states is None else output_hidden_states
        output_attentions = self.config.output_attentions if output_attentions is None else output_attentions
        outputs = self.segformer.encoder(pixel_values, output_attentions=output_attentions,
                                         output_hidden_states=True, return_dict=True)
        logits = self.decode_head(outputs.hidden_states, input_size=pixel_values.shape[-2:],
                                  high_resolution_features=outputs.high_resolution_features)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(F.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False),
                                   labels, ignore_index=self.config.semantic_loss_ignore_index)
        if not return_dict:
            output = (logits,)
            if output_hidden_states:
                output += (outputs.hidden_states,)
            if output_attentions:
                output += (outputs.attentions,)
            return (loss,) + output if loss is not None else output
        return SemanticSegmenterOutput(loss=loss, logits=logits,
                                       hidden_states=outputs.hidden_states if output_hidden_states else None,
                                       attentions=outputs.attentions)
