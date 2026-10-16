"""MiT encoder initialization and offline checkpoint reconstruction."""

from __future__ import annotations

import copy
import logging

import torch
from transformers import SegformerConfig, SegformerForImageClassification, SegformerForSemanticSegmentation

from .config import DEFAULTS, default_config, validate_config
from .decoder import DecoderHead, DecoderSegformerForSemanticSegmentation
from .refinement import RefinedDecodeHead


ARCHITECTURE_FIELDS = (
    "module_a", "module_b", "local_kernels", "refinement_stages", "gate_mode",
    "gate_reduction", "gate_init_probability", "residual_init_std",
    "decoder_fusion", "decoder_upsampling", "upsample_channels",
    "freqfusion", "freqfusion_compressed_channels", "freqfusion_lowpass_kernel",
    "freqfusion_highpass_kernel", "freqfusion_resample_groups", "freqfusion_checkpoint",
    "freqfusion_chunk_size",
)


def architecture_from_config(config: dict) -> dict:
    """Canonical, serializable switches; absent fields describe the original model."""
    if not isinstance(config, dict):
        raise ValueError("Model architecture metadata must be a mapping.")
    architecture = {key: copy.deepcopy(config.get(key, DEFAULTS[key])) for key in ARCHITECTURE_FIELDS}
    validation = default_config()
    validation.update(architecture)
    validate_config(validation)
    # Stage order does not change the computation or state-dict layout.
    architecture["refinement_stages"] = sorted(architecture["refinement_stages"])
    return architecture


def checkpoint_architecture(checkpoint: dict) -> dict:
    """Check all saved architecture descriptions rather than guessing from weights."""
    descriptions = []
    if "architecture" in checkpoint:
        descriptions.append(("architecture", checkpoint["architecture"]))
    model_config = checkpoint["model_config"]
    if "local_refinement" in model_config:
        descriptions.append(("model_config.local_refinement", model_config["local_refinement"]))
    saved_config = checkpoint.get("config", {})
    if any(key in saved_config for key in ARCHITECTURE_FIELDS):
        descriptions.append(("config", saved_config))
    if not descriptions:
        return architecture_from_config({})
    name, raw = descriptions[0]
    architecture = architecture_from_config(raw)
    for other_name, other_raw in descriptions[1:]:
        if architecture_from_config(other_raw) != architecture:
            raise ValueError(f"Checkpoint architecture metadata disagree: {name} vs {other_name}.")
    return architecture


def model_architecture(model) -> dict:
    return architecture_from_config(getattr(model.config, "local_refinement", {}))


def checkpoint_model_signature(checkpoint: dict) -> dict:
    """Compare model settings while allowing optional metadata added to old runs."""
    settings = SegformerConfig.from_dict(checkpoint["model_config"]).to_dict()
    for key in ("local_refinement", "architectures", "transformers_version", "_name_or_path"):
        settings.pop(key, None)
    return settings


def model_parameter_counts(model) -> dict:
    """Parameter counts only; no FLOPs or timing estimates are inferred."""
    count = lambda module: sum(parameter.numel() for parameter in module.parameters())
    encoder = count(model.segformer)
    decoder = count(model.decode_head)
    freqfusion = count(model.decode_head.freqfusion_blocks) if (
        isinstance(model.decode_head, DecoderHead) and model.decode_head.freqfusion
    ) else 0
    if isinstance(model.decode_head, RefinedDecodeHead):
        original_decoder = count(model.decode_head.original_head)
        refinement, redesign = decoder - original_decoder, 0
    elif isinstance(model.decode_head, DecoderHead):
        refinement = 0
        if model.decode_head.fusion == "mlp" and model.decode_head.upsampling == "bilinear":
            original_decoder, redesign = decoder - freqfusion, 0
        else:
            # Replacement head, excluding the separately reported FF modules.
            original_decoder, redesign = 0, decoder - freqfusion
    else:
        original_decoder, refinement, redesign = decoder, 0, 0
    return {"total": count(model), "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "encoder": encoder, "decoder": decoder, "original_decoder": original_decoder,
            "refinement": refinement, "decoder_redesign": redesign, "freqfusion": freqfusion}


def _uses_decoder_redesign(architecture: dict) -> bool:
    return architecture["decoder_fusion"] != "mlp" or architecture["decoder_upsampling"] != "bilinear"


def _uses_custom_decoder(architecture: dict) -> bool:
    return _uses_decoder_redesign(architecture) or architecture["freqfusion"]


def _attach_architecture(model, architecture: dict) -> None:
    # Retain the schema-2 metadata key so old A/B checkpoints remain readable.
    # It now describes all optional decoder and refinement switches.
    model.config.local_refinement = copy.deepcopy(architecture)
    if _uses_custom_decoder(architecture):
        with torch.random.fork_rng(devices=[]):
            model.decode_head = DecoderHead(model.decode_head, model.config, architecture)
    elif architecture["module_a"]:
        # Only new CPU modules consume this forked RNG. The original encoder/head
        # initialization and the outer CPU/CUDA training RNG are retained.
        with torch.random.fork_rng(devices=[]):
            model.decode_head = RefinedDecodeHead(model.decode_head, model.config, architecture)


def _new_model(model_config: SegformerConfig, architecture: dict) -> SegformerForSemanticSegmentation:
    model_class = (DecoderSegformerForSemanticSegmentation if _uses_custom_decoder(architecture)
                   else SegformerForSemanticSegmentation)
    return model_class(model_config)


def build_model(config: dict) -> SegformerForSemanticSegmentation:
    architecture = architecture_from_config(config)
    channels = config["input_channels"]
    if config["pretrained"]:
        source = SegformerForImageClassification.from_pretrained(config["pretrained_model"])
        model_config = SegformerConfig.from_dict(source.config.to_dict())
    else:
        source = None
        model_config = SegformerConfig()  # Default MiT-B0 architecture.
    model_config.num_channels = len(channels)
    model_config.num_labels = 2
    model_config.id2label = {0: "background", 1: "target"}
    model_config.label2id = {"background": 0, "target": 1}
    model_config.architectures = ["SegformerForSemanticSegmentation"]
    model = _new_model(model_config, architecture)
    model.config.architectures = [type(model).__name__]
    if source is None:
        _attach_architecture(model, architecture)
        return model

    source_state, target_state = source.state_dict(), model.state_dict()
    copied = set()
    for key, target in target_state.items():
        if not key.startswith("segformer.") or key not in source_state:
            continue
        value = source_state[key]
        if key.endswith("patch_embeddings.0.proj.weight"):
            if value.ndim != 4 or value.shape[1] != 3:
                raise ValueError("The pretrained encoder must have a three-channel RGB input.")
            # GF B,G,R,NIR -> ImageNet B,G,R,mean(RGB) filters.
            filters = {0: value[:, 2], 1: value[:, 1], 2: value[:, 0], 3: value.mean(dim=1)}
            value = torch.stack([filters[channel] for channel in channels], dim=1)
        if target.shape != value.shape:
            raise ValueError(f"Incompatible encoder tensor {key}: {value.shape} vs {target.shape}")
        target.copy_(value)
        copied.add(key)
    missing = {key for key in target_state if key.startswith("segformer.")} - copied
    if missing:
        raise ValueError(f"Incomplete pretrained encoder: {sorted(missing)[:5]}")
    model.load_state_dict(target_state)
    _attach_architecture(model, architecture)
    logging.getLogger("segformer").info(
        "Loaded %s encoder tensors from %s; input=%s; decoder=%s/%s; freqfusion=%s. Decoder initialized randomly.",
        len(copied), config["pretrained_model"], config["input_bands"],
        architecture["decoder_fusion"], architecture["decoder_upsampling"], architecture["freqfusion"],
    )
    return model


def model_from_checkpoint(checkpoint: dict) -> SegformerForSemanticSegmentation:
    architecture = checkpoint_architecture(checkpoint)
    model = _new_model(SegformerConfig.from_dict(checkpoint["model_config"]), architecture)
    _attach_architecture(model, architecture)
    model.load_state_dict(checkpoint["model"])
    return model
