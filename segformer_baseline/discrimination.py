"""Optional foreground/background feedback inside the second MiT stage.

This is an experimental class-reference branch, not a reproduction of
CamoFormer or ProtoSeg. References are estimated separately for each image
from soft predictions; inference never uses masks or a validation memory.
The encoder loop follows the Transformers 4.44.2 SegFormer stage contract
so corrected F2 is both exposed to the decoder and consumed by stage three.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from transformers.modeling_outputs import BaseModelOutput, SemanticSegmenterOutput

from .decoder import DecoderHead, DecoderSegformerForSemanticSegmentation


@dataclass
class DiscriminativeEncoderOutput(BaseModelOutput):
    auxiliary_logits: torch.Tensor | None = None


@dataclass
class DiscriminativeSemanticSegmenterOutput(SemanticSegmenterOutput):
    auxiliary_logits: torch.Tensor | None = None


class ForegroundBackgroundDiscrimination(nn.Module):
    """Compare each location with two image-local soft class references.

    Softmax channels are ordered background, foreground. Spatially normalized
    probabilities aggregate the embedded F2 features into one reference per
    class. Cosine comparisons retrieve class context for each location; a
    small projection predicts a residual in the original feature channels.
    The final projection starts at zero, preserving the pretrained pathway
    initially. Auxiliary CE trains the embedding and coarse classifier from
    the first step. No threshold or hard foreground selection is involved.

    ``auxiliary`` retains the same embedding/coarse head but omits feedback
    layers, allowing auxiliary supervision to be assessed independently.
    """

    def __init__(self, channels: int, width: int = 32, mode: str = "feedback",
                 temperature: float = 0.2) -> None:
        super().__init__()
        if mode not in ("feedback", "auxiliary"):
            raise ValueError("Discrimination mode must be feedback or auxiliary.")
        if channels < 1 or width < 2 or not 0 < temperature < float("inf"):
            raise ValueError("Discrimination dimensions/temperature must be positive.")
        self.mode, self.temperature = mode, float(temperature)
        self.embedding = nn.Sequential(
            nn.Conv2d(channels, width, kernel_size=1, bias=False),
            nn.GroupNorm(1, width),
            nn.GELU(),
        )
        self.coarse_classifier = nn.Conv2d(width, 2, kernel_size=1)
        if mode == "feedback":
            self.refine = nn.Sequential(
                nn.Conv2d(2 * width + 2, width, kernel_size=1, bias=False),
                nn.GroupNorm(1, width),
                nn.GELU(),
            )
            self.residual_projection = nn.Conv2d(width, channels, kernel_size=1)
            nn.init.zeros_(self.residual_projection.weight)
            nn.init.zeros_(self.residual_projection.bias)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedded = self.embedding(features)
        auxiliary_logits = self.coarse_classifier(embedded)
        if self.mode == "auxiliary":
            return features, auxiliary_logits

        batch, width, height, columns = embedded.shape
        # Reductions and normalization use fp32 even under external autocast.
        # Clamp avoids division by zero if a class gets negligible soft mass.
        with torch.autocast(device_type=embedded.device.type, enabled=False):
            pixels = embedded.flatten(2).float()
            probabilities = auxiliary_logits.flatten(2).float().softmax(dim=1)
            weights = probabilities / probabilities.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            references = torch.bmm(weights, pixels.transpose(1, 2))  # B, 2, D
            similarity = torch.bmm(
                F.normalize(references, dim=-1, eps=1e-6),
                F.normalize(pixels, dim=1, eps=1e-6),
            )  # B, 2, H*W; background first
            assignment = (similarity / self.temperature).softmax(dim=1)
            context = torch.bmm(references.transpose(1, 2), assignment)
        context = context.reshape(batch, width, height, columns).to(embedded.dtype)
        comparison = similarity.reshape(batch, 2, height, columns).to(embedded.dtype)
        correction = self.residual_projection(
            self.refine(torch.cat((embedded, context, comparison), dim=1))
        )
        return features + correction, auxiliary_logits


class DiscriminativeEncoder(nn.Module):
    """Reuse the HF encoder parameters, with feedback after stage-two norm.

    Original module names/order remain intact for state-dict compatibility.
    No copy of the original encoder is retained or initialized a second time.
    """

    def __init__(self, original_encoder, model_config, architecture: dict) -> None:
        super().__init__()
        if len(model_config.hidden_sizes) != 4 or not model_config.reshape_last_stage:
            raise ValueError("Encoder discrimination requires four native 4D MiT feature stages.")
        self.config = model_config
        self.patch_embeddings = original_encoder.patch_embeddings
        self.block = original_encoder.block
        self.layer_norm = original_encoder.layer_norm
        self.discrimination = ForegroundBackgroundDiscrimination(
            model_config.hidden_sizes[1], width=architecture["discrimination_channels"],
            mode=architecture["discrimination_mode"],
            temperature=architecture["discrimination_temperature"],
        )

    def forward(self, pixel_values: torch.Tensor, output_attentions: bool = False,
                output_hidden_states: bool = False, return_dict: bool = True):
        # Same patch/block/norm ordering as HF 4.44.2 SegformerEncoder.forward.
        hidden_states = pixel_values
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        auxiliary_logits = None
        batch_size = pixel_values.shape[0]
        for index, (embedding, blocks, norm) in enumerate(
            zip(self.patch_embeddings, self.block, self.layer_norm)
        ):
            hidden_states, height, width = embedding(hidden_states)
            for block in blocks:
                layer_outputs = block(hidden_states, height, width, output_attentions=output_attentions)
                hidden_states = layer_outputs[0]
                if output_attentions:
                    all_self_attentions += (layer_outputs[1],)
            hidden_states = norm(hidden_states)
            hidden_states = hidden_states.reshape(batch_size, height, width, -1).permute(0, 3, 1, 2).contiguous()
            if index == 1:
                hidden_states, auxiliary_logits = self.discrimination(hidden_states)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
        if not return_dict:
            return tuple(item for item in (hidden_states, all_hidden_states, all_self_attentions,
                                          auxiliary_logits) if item is not None)
        return DiscriminativeEncoderOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states,
            attentions=all_self_attentions, auxiliary_logits=auxiliary_logits,
        )


class DiscriminationSegformerForSemanticSegmentation(DecoderSegformerForSemanticSegmentation):
    """Expose coarse logits alongside the unchanged final prediction API."""

    def forward(self, pixel_values: torch.Tensor, labels: torch.Tensor | None = None,
                output_attentions: bool | None = None, output_hidden_states: bool | None = None,
                return_dict: bool | None = None):
        return_dict = self.config.use_return_dict if return_dict is None else return_dict
        output_hidden_states = self.config.output_hidden_states if output_hidden_states is None else output_hidden_states
        output_attentions = self.config.output_attentions if output_attentions is None else output_attentions
        if pixel_values.ndim != 4:
            raise ValueError("pixel_values must have shape (batch, channels, height, width).")
        # SegformerModel would discard the additional encoder output field.
        outputs = self.segformer.encoder(
            pixel_values, output_attentions=output_attentions,
            output_hidden_states=True, return_dict=True,
        )
        if isinstance(self.decode_head, DecoderHead):
            logits = self.decode_head(outputs.hidden_states, input_size=pixel_values.shape[-2:])
        else:
            logits = self.decode_head(outputs.hidden_states)
        loss = None
        if labels is not None:
            if self.config.num_labels != 2:
                raise ValueError("Encoder discrimination requires background/foreground labels.")
            size = labels.shape[-2:]
            main_logits = F.interpolate(logits, size=size, mode="bilinear", align_corners=False)
            coarse_logits = F.interpolate(outputs.auxiliary_logits, size=size,
                                          mode="bilinear", align_corners=False)
            loss = F.cross_entropy(main_logits, labels, ignore_index=self.config.semantic_loss_ignore_index)
            weight = self.config.local_refinement["discrimination_aux_weight"]
            if weight:
                loss = loss + weight * F.cross_entropy(
                    coarse_logits, labels, ignore_index=self.config.semantic_loss_ignore_index,
                )
        if not return_dict:
            output = (logits,)
            if output_hidden_states:
                output += (outputs.hidden_states,)
            if output_attentions:
                output += (outputs.attentions,)
            output += (outputs.auxiliary_logits,)
            return (loss,) + output if loss is not None else output
        return DiscriminativeSemanticSegmenterOutput(
            loss=loss, logits=logits, hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions, auxiliary_logits=outputs.auxiliary_logits,
        )
