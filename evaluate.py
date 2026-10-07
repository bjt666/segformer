"""Evaluate a saved SegFormer checkpoint on the held-out test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import SegformerConfig, SegformerForSemanticSegmentation

from train import PairedTiffDataset, evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("dataset"))
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Path to outputs/.../best.pt")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    mean = np.asarray(checkpoint["normalization_mean"], dtype=np.float32)
    std = np.asarray(checkpoint["normalization_std"], dtype=np.float32)
    test_set = PairedTiffDataset(args.data_root, "test", mean, std, augment=False)
    loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers,
                        pin_memory=args.device.startswith("cuda"),
                        persistent_workers=args.num_workers > 0)

    config = SegformerConfig.from_dict(checkpoint["model_config"])
    model = SegformerForSemanticSegmentation(config)
    model.load_state_dict(checkpoint["model"])
    device = torch.device(args.device)
    model.to(device)
    metrics = evaluate(model, loader, device)
    print(json.dumps({"split": "test", "num_images": len(test_set), **metrics}, indent=2))


if __name__ == "__main__":
    main()
