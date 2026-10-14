"""Optional local residual refinement before the original SegFormer decoder.

The encoder and the original MLP decoder remain unchanged. Module A supplies
local residual candidates at the selected shallow stages. Module B optionally
selects each candidate with an independent spatial sigmoid gate, using either
shallow information alone or shallow information plus deep context.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.nn import functional as F

if TYPE_CHECKING:
    from transformers import SegformerConfig


class _LocalResidual(nn.Module):
    """One depthwise/local branch with a small, trainable initial residual."""

    def __init__(self, channels: int, kernel_size: int, init_std: float) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            channels, channels, kernel_size, padding=kernel_size // 2, groups=channels
        )
        self.activation = nn.GELU()
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1)
        nn.init.normal_(self.pointwise.weight, mean=0.0, std=init_std)
        nn.init.zeros_(self.pointwise.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.activation(self.depthwise(features)))


class _StageRefinement(nn.Module):
    """Independent local candidates and optional spatial gates for one stage."""

    def __init__(
        self, channels: int, deep_channels: tuple[int, int], architecture: dict
    ) -> None:
        super().__init__()
        kernels = architecture["local_kernels"]
        self.branches = nn.ModuleList(
            _LocalResidual(channels, kernel, architecture["residual_init_std"])
            for kernel in kernels
        )
        self.module_b = architecture["module_b"]
        self.gate_mode = architecture["gate_mode"]

        if not self.module_b:
            return

        # Gate/context initialization must not shift the random initialization
        # of the next stage's local candidates in A-only vs A+B comparisons.
        with torch.random.fork_rng(devices=[]):
            self._build_gate(channels, deep_channels, architecture)

    def _build_gate(self, channels: int, deep_channels: tuple[int, int], architecture: dict) -> None:
        kernels = architecture["local_kernels"]
        gate_input_channels = channels * (1 + len(kernels))
        if self.gate_mode == "deep_shallow":
            self.deep_projections = nn.ModuleList(
                nn.Conv2d(deep_channel, channels, kernel_size=1)
                for deep_channel in deep_channels
            )
            self.deep_fusion = nn.Conv2d(channels * 2, channels, kernel_size=1)
            gate_input_channels += channels

        bottleneck_channels = max(1, channels // architecture["gate_reduction"])
        self.gate = nn.Sequential(
            nn.Conv2d(gate_input_channels, bottleneck_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(bottleneck_channels, len(kernels), kernel_size=3, padding=1),
        )
        # Each branch starts with the same probability. For the default three
        # branches and p=1/3, the initial weighted sum equals A-only averaging.
        # Sigmoids are independent: all branches can be suppressed together.
        nn.init.zeros_(self.gate[-1].weight)
        probability = architecture["gate_init_probability"]
        nn.init.constant_(self.gate[-1].bias, math.log(probability / (1.0 - probability)))

    def forward(
        self, features: torch.Tensor, deep_features: tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        residuals = [branch(features) for branch in self.branches]
        if not self.module_b:
            return features + torch.stack(residuals, dim=0).mean(dim=0)

        gate_inputs = [features, *residuals]
        if self.gate_mode == "deep_shallow":
            aligned_deep = []
            for projection, deep_feature in zip(self.deep_projections, deep_features):
                if deep_feature.ndim != 4:
                    raise ValueError(
                        "Deep-shallow refinement requires 4D F3 and F4 feature maps; "
                        "set SegformerConfig.reshape_last_stage=True."
                    )
                projected = projection(deep_feature)
                aligned_deep.append(
                    F.interpolate(
                        projected, size=features.shape[-2:], mode="bilinear", align_corners=False
                    )
                )
            gate_inputs.append(self.deep_fusion(torch.cat(aligned_deep, dim=1)))

        gates = torch.sigmoid(self.gate(torch.cat(gate_inputs, dim=1)))
        weighted_residuals = [
            residual * gates[:, index : index + 1]
            for index, residual in enumerate(residuals)
        ]
        return features + torch.stack(weighted_residuals, dim=0).sum(dim=0)


class RefinedDecodeHead(nn.Module):
    """Refine shallow features, then delegate to the original decode head.

    Stage indices in ``architecture['refinement_stages']`` are one-based and
    limited to F1/F2. Native four-dimensional feature maps are required for
    those stages. Deep-shallow gating additionally requires native 4D F3/F4
    (the default for MiT-B0). A-only and shallow gating leave F3/F4 untouched,
    so the original decoder can still handle its own supported last-stage
    representation. All added parameters belong to ``model.decode_head``.
    """

    def __init__(
        self, original_head: nn.Module, model_config: SegformerConfig, architecture: dict
    ) -> None:
        super().__init__()
        if not architecture["module_a"]:
            raise ValueError("RefinedDecodeHead requires module_a=True.")
        if architecture["gate_mode"] not in {"deep_shallow", "shallow"}:
            raise ValueError("gate_mode must be 'deep_shallow' or 'shallow'.")
        hidden_sizes = tuple(model_config.hidden_sizes)
        if len(hidden_sizes) != 4:
            raise ValueError("Local refinement currently supports four-stage MiT encoders only.")
        stages = architecture["refinement_stages"]
        if not stages or len(stages) != len(set(stages)) or any(stage not in {1, 2} for stage in stages):
            raise ValueError("refinement_stages must contain unique stage indices from [1, 2].")
        kernels = architecture["local_kernels"]
        if not kernels or any(
            type(kernel) is not int or not 3 <= kernel <= 31 or kernel % 2 == 0
            for kernel in kernels
        ):
            raise ValueError("local_kernels must contain odd integers from 3 to 31.")
        if architecture["gate_reduction"] < 1:
            raise ValueError("gate_reduction must be positive.")
        if not 0.0 < architecture["gate_init_probability"] < 1.0:
            raise ValueError("gate_init_probability must be strictly between zero and one.")
        if not math.isfinite(architecture["residual_init_std"]) or architecture["residual_init_std"] < 0:
            raise ValueError("residual_init_std must be finite and nonnegative.")
        if (
            architecture["module_b"]
            and architecture["gate_mode"] == "deep_shallow"
            and not model_config.reshape_last_stage
        ):
            raise ValueError(
                "Deep-shallow refinement requires SegformerConfig.reshape_last_stage=True; "
                "a flattened last-stage feature cannot be aligned without its native spatial shape."
            )

        self.original_head = original_head
        self.refinements = nn.ModuleDict(
            {
                str(stage): _StageRefinement(hidden_sizes[stage - 1], hidden_sizes[2:4], architecture)
                for stage in stages
            }
        )

    def forward(self, encoder_hidden_states: tuple[torch.Tensor, ...] | list[torch.Tensor]) -> torch.Tensor:
        if len(encoder_hidden_states) != 4:
            raise ValueError("Local refinement requires exactly four encoder feature stages.")
        features = list(encoder_hidden_states)
        # Read deep context from the unchanged encoder features, independently
        # for each shallow stage. Do not mutate the encoder's output container.
        deep_features = (encoder_hidden_states[2], encoder_hidden_states[3])
        for stage_key, refinement in self.refinements.items():
            stage_index = int(stage_key) - 1
            if features[stage_index].ndim != 4:
                raise ValueError(f"Refinement stage F{stage_key} must be a native 4D feature map.")
            features[stage_index] = refinement(features[stage_index], deep_features)
        return self.original_head(features)
