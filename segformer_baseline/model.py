"""MiT encoder initialization and offline checkpoint reconstruction."""

from __future__ import annotations

import copy
import logging

import torch
from transformers import SegformerConfig, SegformerForImageClassification, SegformerForSemanticSegmentation

from .config import DEFAULTS, default_config, validate_config
from .refinement import RefinedDecodeHead


ARCHITECTURE_FIELDS = (
    "module_a", "module_b", "local_kernels", "refinement_stages", "gate_mode",
    "gate_reduction", "gate_init_probability", "residual_init_std",
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
    original_head = (model.decode_head.original_head if isinstance(model.decode_head, RefinedDecodeHead)
                     else model.decode_head)
    original_decoder = count(original_head)
    return {"total": count(model), "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "encoder": encoder, "decoder": decoder, "original_decoder": original_decoder,
            "refinement": decoder - original_decoder}


def _attach_refinement(model, architecture: dict) -> None:
    model.config.local_refinement = copy.deepcopy(architecture)
    if architecture["module_a"]:
        # Only new CPU modules consume this forked RNG. The original encoder/head
        # initialization and the outer CPU/CUDA training RNG are retained.
        with torch.random.fork_rng(devices=[]):
            model.decode_head = RefinedDecodeHead(model.decode_head, model.config, architecture)


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
    model = SegformerForSemanticSegmentation(model_config)
    if source is None:
        _attach_refinement(model, architecture)
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
    _attach_refinement(model, architecture)
    logging.getLogger("segformer").info(
        "Loaded %s encoder tensors from %s; input=%s; original MLP decoder initialized randomly.",
        len(copied), config["pretrained_model"], config["input_bands"],
    )
    return model


def model_from_checkpoint(checkpoint: dict) -> SegformerForSemanticSegmentation:
    architecture = checkpoint_architecture(checkpoint)
    model = SegformerForSemanticSegmentation(SegformerConfig.from_dict(checkpoint["model_config"]))
    _attach_refinement(model, architecture)
    model.load_state_dict(checkpoint["model"])
    return model
