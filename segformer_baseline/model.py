"""MiT encoder initialization and offline checkpoint reconstruction."""

from __future__ import annotations

import logging

import torch
from transformers import SegformerConfig, SegformerForImageClassification, SegformerForSemanticSegmentation


def build_model(config: dict) -> SegformerForSemanticSegmentation:
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
    model = SegformerForSemanticSegmentation(model_config)
    if source is None:
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
    logging.getLogger("segformer").info(
        "Loaded %s encoder tensors from %s; input=%s; original MLP decoder initialized randomly.",
        len(copied), config["pretrained_model"], config["input_bands"],
    )
    return model


def model_from_checkpoint(checkpoint: dict) -> SegformerForSemanticSegmentation:
    model = SegformerForSemanticSegmentation(SegformerConfig.from_dict(checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model"])
    return model
