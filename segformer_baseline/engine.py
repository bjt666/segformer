"""Training and evaluation loops, with terminal bars and periodic log messages."""

from __future__ import annotations

import logging
import math
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .metrics import confusion_matrix, metrics_from_confusion


def resized_logits(model, images: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    logits = model(pixel_values=images).logits
    return F.interpolate(logits, size=size, mode="bilinear", align_corners=False)


def train_epoch(model, loader, optimizer, criterion, device: torch.device,
                epoch: int, total_epochs: int, log_every: int) -> dict:
    logger = logging.getLogger("segformer")
    model.train()
    started = time.monotonic()
    total_loss = 0.0
    num_images = 0
    bar = tqdm(loader, desc=f"Train {epoch}/{total_epochs}", disable=not sys.stderr.isatty())
    for step, batch in enumerate(bar, start=1):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = resized_logits(model, images, masks.shape[-2:])
        loss = criterion(logits, masks)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite training loss at epoch {epoch}, batch {step}.")
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        num_images += images.size(0)
        bar.set_postfix(loss=f"{total_loss / num_images:.5f}", lr=f"{optimizer.param_groups[0]['lr']:.2g}")
        if step == 1 or step % log_every == 0 or step == len(loader):
            logger.info(
                "Train epoch %s/%s batch %s/%s | loss %.6f | encoder lr %.3g | decoder lr %.3g | %.1fs",
                epoch, total_epochs, step, len(loader), total_loss / num_images,
                optimizer.param_groups[0]["lr"], optimizer.param_groups[1]["lr"], time.monotonic() - started,
            )
    return {"loss": total_loss / num_images, "seconds": time.monotonic() - started}


@torch.inference_mode()
def evaluate_model(model, loader, device: torch.device, criterion=None,
                   description: str = "Evaluate", log_every: int = 20,
                   collect_per_image: bool = True, sample_limit: int = 0) -> dict:
    logger = logging.getLogger("segformer")
    model.eval()
    started = time.monotonic()
    confusion = np.zeros((2, 2), dtype=np.int64)
    per_image, samples = [], []
    total_loss, num_images = 0.0, 0
    bar = tqdm(loader, desc=description, disable=not sys.stderr.isatty())
    for step, batch in enumerate(bar, start=1):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        logits = resized_logits(model, images, masks.shape[-2:])
        if criterion is not None:
            loss_value = criterion(logits, masks).item()
            if not math.isfinite(loss_value):
                raise FloatingPointError(f"Nonfinite loss in {description}, batch {step}.")
            total_loss += loss_value * images.size(0)
        prediction = logits.argmax(dim=1).cpu().numpy().astype(np.uint8)
        target = masks.cpu().numpy()
        for index, name in enumerate(batch["name"]):
            matrix = confusion_matrix(target[index], prediction[index])
            confusion += matrix
            if collect_per_image:
                per_image.append({"name": name, **metrics_from_confusion(matrix)})
            if len(samples) < sample_limit:
                samples.append({"name": name, "image_path": batch["image_path"][index],
                                "prediction": prediction[index], "target": target[index]})
        num_images += images.size(0)
        if criterion is not None:
            bar.set_postfix(loss=f"{total_loss / num_images:.5f}")
        if step == 1 or step % log_every == 0 or step == len(loader):
            logger.info("%s batch %s/%s | %s images | %.1fs", description, step, len(loader),
                        num_images, time.monotonic() - started)
    return {"metrics": metrics_from_confusion(confusion), "confusion": confusion,
            "per_image": per_image, "samples": samples, "num_images": num_images,
            "loss": total_loss / num_images if criterion is not None else None,
            "seconds": time.monotonic() - started}
