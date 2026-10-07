"""Paired TIFF datasets, train-only normalization and content manifests."""

from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import tifffile
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


DEFAULT_CHANNELS = (3, 2, 1)
DEFAULT_BANDS = ("NIR", "R", "G")


def image_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {directory}")
    paths = sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in {".tif", ".tiff"})
    if not paths:
        raise FileNotFoundError(f"No TIFF images found in {directory}")
    return paths


def read_image(path: Path) -> np.ndarray:
    image = tifffile.imread(path).astype(np.float32)
    if image.ndim != 3 or image.shape[-1] != 4:
        raise ValueError(f"Expected an HxWx4 TIFF in B,G,R,NIR order: {path}; got {image.shape}")
    if not np.isfinite(image).all():
        raise ValueError(f"Nonfinite image values: {path}")
    return image


def read_mask(path: Path, shape: tuple[int, int]) -> np.ndarray:
    mask = tifffile.imread(path)
    if mask.ndim != 2 or mask.shape != shape:
        raise ValueError(f"Expected mask shape {shape}, got {mask.shape}: {path}")
    if not np.isin(mask, [0, 1, 255]).all():
        raise ValueError(f"Binary masks must contain only 0, 1 or 255: {path}")
    return (mask > 0).astype(np.int64)


def normalize_image(image: np.ndarray, channels: tuple[int, ...], mean: np.ndarray,
                    std: np.ndarray) -> torch.Tensor:
    selected = image[..., list(channels)]
    selected = (selected - np.asarray(mean, dtype=np.float32).reshape(1, 1, -1)) / np.maximum(
        np.asarray(std, dtype=np.float32).reshape(1, 1, -1), 1e-6
    )
    return torch.from_numpy(np.ascontiguousarray(selected.transpose(2, 0, 1), dtype=np.float32))


class PairedTiffDataset(Dataset):
    def __init__(self, root: Path, split: str, mean: np.ndarray, std: np.ndarray,
                 channels: tuple[int, ...] = DEFAULT_CHANNELS, augment: bool = False) -> None:
        self.image_dir = root / "images" / split
        self.mask_dir = root / "masks" / split
        images = {p.name: p for p in image_files(self.image_dir)}
        masks = {p.name: p for p in image_files(self.mask_dir)}
        if images.keys() != masks.keys():
            raise ValueError(
                f"Unpaired files in {split}: missing masks={sorted(images.keys() - masks.keys())[:5]}, "
                f"missing images={sorted(masks.keys() - images.keys())[:5]}"
            )
        self.names = sorted(images)
        self.channels = tuple(channels)
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        if self.mean.shape != (len(channels),) or self.std.shape != (len(channels),):
            raise ValueError("Normalization statistics must match the number of selected channels.")
        if not np.isfinite(self.mean).all() or not np.isfinite(self.std).all() or (self.std <= 0).any():
            raise ValueError("Normalization statistics must be finite, with positive standard deviations.")
        self.augment = augment

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, index: int) -> dict:
        name = self.names[index]
        image = read_image(self.image_dir / name)
        mask = read_mask(self.mask_dir / name, image.shape[:2])
        if self.augment:
            if random.random() < 0.5:
                image, mask = image[:, ::-1, :], mask[:, ::-1]
            if random.random() < 0.5:
                image, mask = image[::-1, :, :], mask[::-1, :]
            turns = random.randrange(4)
            if turns:
                image = np.rot90(image, turns, axes=(0, 1))
                mask = np.rot90(mask, turns)
        return {
            "image": normalize_image(image, self.channels, self.mean, self.std),
            "mask": torch.from_numpy(np.ascontiguousarray(mask, dtype=np.int64)),
            "name": name, "image_path": str(self.image_dir / name),
        }


def compute_train_stats(root: Path, channels: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    paths = image_files(root / "images" / "train")
    total = np.zeros(len(channels), dtype=np.float64)
    total_sq = np.zeros(len(channels), dtype=np.float64)
    count = 0
    for path in tqdm(paths, desc="Train normalization", disable=not sys.stderr.isatty()):
        image = read_image(path)[..., list(channels)].astype(np.float64)
        total += image.sum(axis=(0, 1))
        total_sq += np.square(image).sum(axis=(0, 1))
        count += image.shape[0] * image.shape[1]
    mean = total / count
    std = np.sqrt(np.maximum(total_sq / count - np.square(mean), 1e-12))
    return mean.astype(np.float32), std.astype(np.float32)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_manifest(root: Path, splits: tuple[str, ...]) -> dict:
    """Record names and content hashes; the fingerprint is independent of root path."""
    records = []
    for split in splits:
        for kind in ("images", "masks"):
            for path in image_files(root / kind / split):
                records.append({"path": path.relative_to(root).as_posix(),
                                "size": path.stat().st_size, "sha256": file_sha256(path)})
    fingerprint = hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()
    return {"algorithm": "sha256", "splits": list(splits), "fingerprint": fingerprint, "files": records}
