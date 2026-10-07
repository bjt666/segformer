"""Train a SegFormer-B0 binary segmentation baseline on paired TIFF files."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import tifffile
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    SegformerConfig,
    SegformerForImageClassification,
    SegformerForSemanticSegmentation,
)

# TIFFs store the GF multispectral bands as B, G, R, NIR. Use a vegetation
# false-color input with the displayed channels [NIR, R, G].
INPUT_CHANNELS = (3, 2, 1)
INPUT_BANDS = ("NIR", "R", "G")


class PairedTiffDataset(Dataset):
    """Loads images/<split> and masks/<split>, matching files by name."""

    def __init__(self, root: Path, split: str, mean: np.ndarray, std: np.ndarray,
                 augment: bool = False) -> None:
        self.image_dir = root / "images" / split
        self.mask_dir = root / "masks" / split
        images = {p.name: p for p in self.image_dir.glob("*.tif")}
        masks = {p.name: p for p in self.mask_dir.glob("*.tif")}
        missing_masks = sorted(images.keys() - masks.keys())
        missing_images = sorted(masks.keys() - images.keys())
        if missing_masks or missing_images:
            raise ValueError(
                f"Unpaired files in {split}: missing masks={missing_masks[:5]}, "
                f"missing images={missing_images[:5]}"
            )
        self.names = sorted(images)
        if not self.names:
            raise FileNotFoundError(f"No .tif images found in {self.image_dir}")
        self.mean = mean.astype(np.float32).reshape(1, 1, -1)
        self.std = np.maximum(std.astype(np.float32), 1e-6).reshape(1, 1, -1)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        name = self.names[index]
        image = tifffile.imread(self.image_dir / name).astype(np.float32)
        mask = tifffile.imread(self.mask_dir / name)
        if image.ndim != 3 or image.shape[-1] != 4:
            raise ValueError(f"Expected HxWx4 image, got {image.shape}: {name}")
        if mask.ndim != 2 or mask.shape != image.shape[:2]:
            raise ValueError(f"Expected matching HxW mask, got {mask.shape}: {name}")

        image = image[..., INPUT_CHANNELS]
        image = (image - self.mean) / self.std
        # Dataset labels are 0 (background) and 255 (foreground).
        mask = (mask > 0).astype(np.int64)

        if self.augment:
            if random.random() < 0.5:
                image, mask = image[:, ::-1, :], mask[:, ::-1]
            if random.random() < 0.5:
                image, mask = image[::-1, :, :], mask[::-1, :]
            turns = random.randrange(4)
            if turns:
                image, mask = np.rot90(image, turns, axes=(0, 1)), np.rot90(mask, turns)

        image = np.ascontiguousarray(image.transpose(2, 0, 1), dtype=np.float32)
        mask = np.ascontiguousarray(mask, dtype=np.int64)
        return torch.from_numpy(image), torch.from_numpy(mask)


def compute_train_stats(root: Path) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-band pixel mean/std from training images only."""
    paths = sorted((root / "images" / "train").glob("*.tif"))
    if not paths:
        raise FileNotFoundError(f"No training TIFF images found in {root / 'images' / 'train'}")
    count = 0
    total = np.zeros(4, dtype=np.float64)
    total_sq = np.zeros(4, dtype=np.float64)
    for path in tqdm(paths, desc="Computing train normalization", leave=False):
        image = tifffile.imread(path).astype(np.float64)
        if image.ndim != 3 or image.shape[-1] != 4:
            raise ValueError(f"Expected HxWx4 image, got {image.shape}: {path}")
        total += image.sum(axis=(0, 1))
        total_sq += np.square(image).sum(axis=(0, 1))
        count += image.shape[0] * image.shape[1]
    mean = total / count
    std = np.sqrt(np.maximum(total_sq / count - np.square(mean), 1e-12))
    return mean[list(INPUT_CHANNELS)].astype(np.float32), std[list(INPUT_CHANNELS)].astype(np.float32)


def build_model(pretrained: bool, checkpoint: str, num_labels: int = 2) -> SegformerForSemanticSegmentation:
    """Create a 3-channel NIR-R-G SegFormer and optionally load an ImageNet encoder."""
    if not pretrained:
        config = SegformerConfig(num_channels=3, num_labels=num_labels)
        return SegformerForSemanticSegmentation(config)

    source = SegformerForImageClassification.from_pretrained(checkpoint)
    config = SegformerConfig.from_pretrained(checkpoint)
    config.num_channels = 3
    config.num_labels = num_labels
    config.id2label = {0: "background", 1: "target"}
    config.label2id = {"background": 0, "target": 1}
    model = SegformerForSemanticSegmentation(config)

    source_state = source.state_dict()
    target_state = model.state_dict()
    copied = 0
    adapted_input = False
    for key, value in source_state.items():
        if key not in target_state:
            continue
        target = target_state[key]
        if key.endswith("patch_embeddings.0.proj.weight") and value.ndim == 4 \
                and value.shape[1] == 3 and target.shape[1] == 3:
            # Input order is NIR, R, G. Initialize NIR with the mean RGB filter,
            # then map red and green to their corresponding ImageNet filters.
            target.copy_(torch.stack((value.mean(dim=1), value[:, 0], value[:, 1]), dim=1))
            copied += 1
            adapted_input = True
        elif value.shape == target.shape:
            target.copy_(value)
            copied += 1
    if not adapted_input:
        raise RuntimeError("Could not find/adapt the first 3-to-4-channel patch projection")
    model.load_state_dict(target_state)
    print(f"Loaded {copied} matching ImageNet encoder tensors from {checkpoint}; "
          "the NIR-R-G input projection was adapted and the segmentation head is newly initialized.")
    return model


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    confusion = torch.zeros((2, 2), dtype=torch.float64, device=device)
    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        logits = model(pixel_values=images).logits
        logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
        predictions = logits.argmax(dim=1)
        indices = masks.reshape(-1) * 2 + predictions.reshape(-1)
        confusion += torch.bincount(indices, minlength=4).reshape(2, 2)

    tn, fp, fn, tp = confusion.reshape(-1)
    eps = 1e-8
    iou = tp / (tp + fp + fn + eps)
    dice = (2 * tp) / (2 * tp + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
    return {"foreground_iou": iou.item(), "dice": dice.item(),
            "precision": precision.item(), "recall": recall.item(),
            "pixel_accuracy": accuracy.item()}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/segformer-b0"))
    parser.add_argument("--pretrained-model", default="nvidia/mit-b0",
                        help="ImageNet-1K SegFormer encoder ID or local checkpoint directory")
    parser.add_argument("--no-pretrained", action="store_true",
                        help="Initialize the whole model from scratch")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--encoder-lr", type=float, default=6e-5)
    parser.add_argument("--decoder-lr", type=float, default=6e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch-size must be positive")
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mean, std = compute_train_stats(args.data_root)
    print(f"Train-only per-band normalization: mean={mean.tolist()}, std={std.tolist()}")

    train_set = PairedTiffDataset(args.data_root, "train", mean, std, augment=True)
    val_set = PairedTiffDataset(args.data_root, "val", mean, std, augment=False)
    common: dict[str, Any] = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": args.device.startswith("cuda"),
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(train_set, shuffle=True, drop_last=False, **common)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=False, **common)

    device = torch.device(args.device)
    model = build_model(not args.no_pretrained, args.pretrained_model).to(device)
    optimizer = torch.optim.AdamW([
        {"params": model.segformer.parameters(), "lr": args.encoder_lr},
        {"params": model.decode_head.parameters(), "lr": args.decoder_lr},
    ], weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()
    best_iou = -1.0
    history_path = args.output_dir / "metrics.jsonl"
    run_config = vars(args).copy()
    run_config["data_root"] = str(args.data_root.resolve())
    run_config["output_dir"] = str(args.output_dir.resolve())
    run_config["normalization_mean"] = mean.tolist()
    run_config["normalization_std"] = std.tolist()
    run_config["input_bands"] = list(INPUT_BANDS)
    (args.output_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2), encoding="utf-8"
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(pixel_values=images).logits
            logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        metrics = evaluate(model, val_loader, device)
        scheduler.step()
        record = {"epoch": epoch, "train_loss": running_loss / len(train_set), **metrics,
                  "encoder_lr": optimizer.param_groups[0]["lr"],
                  "decoder_lr": optimizer.param_groups[1]["lr"]}
        with history_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        print(json.dumps(record))
        if metrics["foreground_iou"] > best_iou:
            best_iou = metrics["foreground_iou"]
            torch.save({"model": model.state_dict(), "model_config": model.config.to_dict(),
                        "epoch": epoch,
                        "metrics": metrics, "config": run_config,
                        "normalization_mean": mean.tolist(),
                        "normalization_std": std.tolist()}, args.output_dir / "best.pt")

    print(f"Best validation foreground IoU: {best_iou:.6f}")


if __name__ == "__main__":
    main()
