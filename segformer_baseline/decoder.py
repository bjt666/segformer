"""Optional feature fusion and feature upsampling for the SegFormer decoder.

The original Hugging Face head remains the baseline. ``DecoderHead`` adopts
only the original submodules used by the selected decoder, without keeping a
reference to the discarded head. Both fusion choices stop at the native F1
resolution; the shared dropout is applied once, immediately before the final
classifier, after optional learned feature upsampling.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.nn import functional as F
from transformers import SegformerForSemanticSegmentation
from transformers.modeling_outputs import SemanticSegmenterOutput

from .freqfusion import FreqFusion

if TYPE_CHECKING:
    from transformers import SegformerConfig


class _ConvBnRelu(nn.Module):
    """A spatial refinement block with no change in feature resolution."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.batch_norm = nn.BatchNorm2d(channels)
        self.activation = nn.ReLU()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.activation(self.batch_norm(self.conv(features)))


class _ProgressiveFusion(nn.Module):
    """Fuse F4 into F3, F2, then F1 at each shallower native resolution."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleDict(
            {stage: _ConvBnRelu(width) for stage in ("f3", "f2", "f1")}
        )

    def forward(
        self, projected_features: list[torch.Tensor], freqfusion_blocks: nn.ModuleDict | None = None
    ) -> torch.Tensor:
        features = projected_features[-1]
        for (stage, block), shallow in zip(self.blocks.items(), reversed(projected_features[:-1])):
            if freqfusion_blocks is None:
                features = F.interpolate(
                    features, size=shallow.shape[-2:], mode="bilinear", align_corners=False
                )
            else:
                shallow, features = freqfusion_blocks[stage](shallow, features)
            features = block(features + shallow)
        return features


class _LearnedUpsampling(nn.Module):
    """Restore features to the actual input size before classification.

    The intermediate target is ceil(input height/2), ceil(input width/2).
    Native F1 dimensions are deliberately not used to infer either target,
    because encoder strides round spatial dimensions for odd sized inputs.
    """

    def __init__(self, width: int, channels: tuple[int, int]) -> None:
        super().__init__()
        self.reduce_half = nn.Conv2d(width, channels[0], kernel_size=1)
        self.refine_half = _ConvBnRelu(channels[0])
        self.reduce_full = nn.Conv2d(channels[0], channels[1], kernel_size=1)
        self.refine_full = _ConvBnRelu(channels[1])

    def forward(
        self, features: torch.Tensor, input_size: tuple[int, int] | torch.Size
    ) -> torch.Tensor:
        height, width = input_size
        features = self.reduce_half(features)
        features = F.interpolate(
            features,
            size=((height + 1) // 2, (width + 1) // 2),
            mode="bilinear",
            align_corners=False,
        )
        features = self.refine_half(features)
        features = self.reduce_full(features)
        features = F.interpolate(
            features, size=input_size, mode="bilinear", align_corners=False
        )
        return self.refine_full(features)


class DecoderHead(nn.Module):
    """Reuse the original stage projections with selectable decoder modules.

    ``linear_c`` always reuses the four original MLP projections. MLP fusion
    reuses ``linear_fuse``, ``batch_norm`` and ``activation`` in the original
    F4/F3/F2/F1 concatenation order. Progressive fusion instead registers only
    its three new spatial blocks. ``dropout`` is the original dropout module
    and is called once before ``classifier`` for every decoder combination.

    Bilinear mode keeps the original classifier and returns logits at F1
    resolution for the existing engine to resize. Learned mode registers two
    feature recovery stages and a classifier at the final recovery width; it
    requires the actual input image dimensions and returns full input logits.
    Optional FreqFusion changes the cross-scale alignment of either fusion
    pathway. In MLP mode it recursively aligns and concatenates all projected
    branches, retaining the original final fusion and classifier layers.
    Fusion width is always ``model_config.decoder_hidden_size``.
    """

    def __init__(
        self, original_head: nn.Module, model_config: SegformerConfig, architecture: dict
    ) -> None:
        super().__init__()
        self.fusion = architecture.get("decoder_fusion", "mlp")
        self.upsampling = architecture.get("decoder_upsampling", "bilinear")
        self.freqfusion = architecture.get("freqfusion", False)
        if self.fusion not in {"mlp", "progressive"}:
            raise ValueError("decoder_fusion must be 'mlp' or 'progressive'.")
        if self.upsampling not in {"bilinear", "learned"}:
            raise ValueError("decoder_upsampling must be 'bilinear' or 'learned'.")
        hidden_sizes = tuple(model_config.hidden_sizes)
        if len(hidden_sizes) != 4 or len(original_head.linear_c) != 4:
            raise ValueError("DecoderHead requires exactly four MiT encoder feature stages.")
        if not model_config.reshape_last_stage:
            raise ValueError(
                "DecoderHead requires native 4D F1/F2/F3/F4 feature maps; set "
                "SegformerConfig.reshape_last_stage=True. Flattened F4 is unsupported."
            )
        width = model_config.decoder_hidden_size
        if type(width) is not int or width < 1:
            raise ValueError("decoder_hidden_size must be a positive integer.")
        channels = architecture.get("upsample_channels", [64, 32])
        if (
            not isinstance(channels, (list, tuple))
            or len(channels) != 2
            or any(type(channel) is not int or channel < 1 for channel in channels)
        ):
            raise ValueError("upsample_channels must contain two positive integers.")

        self.hidden_sizes = hidden_sizes
        self.linear_c = original_head.linear_c
        self.dropout = original_head.dropout
        if self.fusion == "mlp":
            self.linear_fuse = original_head.linear_fuse
            self.batch_norm = original_head.batch_norm
            self.activation = original_head.activation
        else:
            # Isolating the fusion RNG means learned recovery starts from the
            # same RNG state for MLP and progressive fusion. It also keeps the
            # progressive initialization equal with or without recovery.
            with torch.random.fork_rng(devices=[]):
                self.progressive_fusion = _ProgressiveFusion(width)

        if self.upsampling == "bilinear":
            self.classifier = original_head.classifier
        else:
            # Keep this construction order fixed across both fusion choices.
            # Reused original projections/fusion are never reinitialized.
            with torch.random.fork_rng(devices=[]):
                self.learned_upsampling = _LearnedUpsampling(width, tuple(channels))
                self.classifier = nn.Conv2d(channels[-1], model_config.num_labels, kernel_size=1)

        if self.freqfusion:
            if width % architecture["freqfusion_resample_groups"]:
                raise ValueError("decoder_hidden_size must be divisible by freqfusion_resample_groups.")
            # Isolate the new modules from the initialization of all reused
            # layers and from either optional decoder redesign component.
            with torch.random.fork_rng(devices=[]):
                self.freqfusion_blocks = nn.ModuleDict({
                    stage: FreqFusion(
                        width, width * (index + 1 if self.fusion == "mlp" else 1),
                        compressed_channels=architecture["freqfusion_compressed_channels"],
                        lowpass_kernel=architecture["freqfusion_lowpass_kernel"],
                        highpass_kernel=architecture["freqfusion_highpass_kernel"],
                        resample_groups=architecture["freqfusion_resample_groups"],
                        checkpoint_reassembly=architecture["freqfusion_checkpoint"],
                        chunk_size=architecture["freqfusion_chunk_size"],
                    ) for index, stage in enumerate(("f3", "f2", "f1"))
                })

    def forward(
        self,
        encoder_hidden_states: tuple[torch.Tensor, ...] | list[torch.Tensor],
        input_size: tuple[int, int] | torch.Size | None = None,
    ) -> torch.Tensor:
        if len(encoder_hidden_states) != 4:
            raise ValueError("DecoderHead requires exactly four encoder feature stages.")
        if self.upsampling == "learned" and (
            input_size is None or len(input_size) != 2 or any(size < 1 for size in input_size)
        ):
            raise ValueError("Learned feature upsampling requires the actual positive input (height, width).")

        projected_features = []
        batch_size = encoder_hidden_states[0].shape[0]
        for index, (features, projection) in enumerate(zip(encoder_hidden_states, self.linear_c)):
            if features.ndim != 4:
                raise ValueError(
                    f"Decoder stage F{index + 1} must be a native 4D feature map; "
                    "flattened F4 is unsupported. Set SegformerConfig.reshape_last_stage=True."
                )
            if features.shape[0] != batch_size or features.shape[1] != self.hidden_sizes[index]:
                raise ValueError(f"Decoder stage F{index + 1} has incompatible batch or channel dimensions.")
            height, width = features.shape[-2:]
            projected = projection(features).permute(0, 2, 1)
            projected_features.append(projected.reshape(batch_size, -1, height, width))

        if self.fusion == "mlp":
            if self.freqfusion:
                # Preserve all four projected branches and the original
                # F4/F3/F2/F1 channel order. Unlike progressive fusion, there
                # are no per-stage additions or extra 3x3 fusion convolutions.
                features = projected_features[-1]
                for module, shallow in zip(
                    self.freqfusion_blocks.values(), reversed(projected_features[:-1])
                ):
                    shallow, features = module(shallow, features)
                    features = torch.cat((features, shallow), dim=1)
            else:
                aligned = [
                    F.interpolate(
                        features,
                        size=encoder_hidden_states[0].shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                    for features in projected_features
                ]
                features = torch.cat(aligned[::-1], dim=1)
            features = self.linear_fuse(features)
            features = self.activation(self.batch_norm(features))
        else:
            features = self.progressive_fusion(
                projected_features, self.freqfusion_blocks if self.freqfusion else None
            )

        if self.upsampling == "learned":
            features = self.learned_upsampling(features, input_size)
        return self.classifier(self.dropout(features))


class DecoderSegformerForSemanticSegmentation(SegformerForSemanticSegmentation):
    """Preserve the HF constructor and pass actual image sizes to DecoderHead.

    A regular or refinement decode head delegates to the original HF forward,
    preserving the baseline and existing A/B behavior. The DecoderHead branch
    always requests encoder stage features internally while exposing hidden
    states and attentions only according to the public output flags.
    """

    def forward(
        self,
        pixel_values: torch.Tensor,
        labels: torch.Tensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> tuple | SemanticSegmenterOutput:
        if not isinstance(self.decode_head, DecoderHead):
            return super().forward(
                pixel_values=pixel_values,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        if pixel_values.ndim != 4:
            raise ValueError("pixel_values must have shape (batch, channels, height, width).")
        if labels is not None and self.config.num_labels <= 1:
            raise ValueError("DecoderHead label loss supports num_labels > 1 with cross entropy only.")

        outputs = self.segformer(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=True,
        )
        logits = self.decode_head(outputs.hidden_states, input_size=pixel_values.shape[-2:])
        loss = None
        if labels is not None:
            loss_logits = F.interpolate(
                logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
            )
            loss = F.cross_entropy(
                loss_logits, labels, ignore_index=self.config.semantic_loss_ignore_index
            )

        if not return_dict:
            output = (logits,)
            if output_hidden_states:
                output += (outputs.hidden_states,)
            if outputs.attentions is not None:
                output += (outputs.attentions,)
            return (loss,) + output if loss is not None else output

        return SemanticSegmenterOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions,
        )
