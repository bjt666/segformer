"""Check one training batch for shapes, gradients and GPU memory; no run is saved.

Random initialization is used unless --load-pretrained is requested. This
checks execution/memory only, not convergence or segmentation accuracy.
"""
from __future__ import annotations

import argparse
import json

import torch

from segformer_baseline.config import default_config, read_config, validate_config
from segformer_baseline.engine import training_losses
from segformer_baseline.model import build_model, model_parameter_counts, optimizer_parameter_groups
from segformer_baseline.utils import resolve_device, set_seed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--device")
    parser.add_argument("--load-pretrained", action="store_true")
    parser.add_argument("--cnn-pretrained-model")
    args = parser.parse_args()
    config = dict(default_config(), **read_config(args.config))
    config["pretrained"] = args.load_pretrained
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.device:
        config["device"] = args.device
    if args.cnn_pretrained_model:
        config["cnn_pretrained_model"] = args.cnn_pretrained_model
    validate_config(config)
    if args.size < 32:
        parser.error("size must be at least 32 for the four-stage encoder")
    device = resolve_device(config["device"])
    set_seed(config["seed"])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    try:
        model = build_model(config).to(device).train()
        optimizer = torch.optim.AdamW(optimizer_parameter_groups(model, config), weight_decay=config["weight_decay"])
        images = torch.randn(config["batch_size"], len(config["input_channels"]), args.size, args.size, device=device)
        masks = torch.randint(0, 2, (config["batch_size"], args.size, args.size), device=device)
        losses = training_losses(model, images, masks, torch.nn.CrossEntropyLoss())
        if not torch.isfinite(losses["loss"]):
            raise FloatingPointError("Nonfinite preflight loss")
        losses["loss"].backward()
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        report = {"status": "passed", "batch_size": config["batch_size"], "image_size": args.size,
                  "device": str(device), "loss": losses["loss"].item(),
                  "parameters": model_parameter_counts(model),
                  "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else None,
                  "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 1024**2 if device.type == "cuda" else None}
        print(json.dumps(report, indent=2))
    except torch.cuda.OutOfMemoryError:
        raise SystemExit("CUDA memory insufficient for this batch. Try a smaller --batch-size in this check; "
                         "use the same setting for the corresponding comparison experiment.")


if __name__ == "__main__":
    main()
