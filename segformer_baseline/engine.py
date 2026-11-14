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

from .contrastive import HardAlgaeContrastiveLoss
from .error_diagnostics import ErrorDiagnostics
from .error_refinement import balanced_error_loss
from .input_scale import forward_with_input_scale, logits_on_grid
from .lovasz import foreground_lovasz_loss
from .metrics import confusion_matrix, metrics_from_confusion


def resized_logits(model, images: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    return logits_on_grid(forward_with_input_scale(model, images).logits, size)


def training_losses(model, images: torch.Tensor, masks: torch.Tensor, criterion, *,
                    contrastive: HardAlgaeContrastiveLoss | None = None,
                    contrastive_weight: float = 0.0, lovasz_weight: float = 0.0) -> dict:
    """Combine segmentation CE and any active training-only supervision."""
    use_contrastive = contrastive is not None and contrastive_weight > 0
    if use_contrastive:
        outputs = forward_with_input_scale(model, images, return_dict=True, return_decoder_features=True)
    else:
        outputs = forward_with_input_scale(model, images)
    logits = outputs.logits
    if logits.shape[-2:] != masks.shape[-2:]:
        logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
    main_loss = criterion(logits, masks)
    auxiliary_logits = getattr(outputs, "auxiliary_logits", None)
    auxiliary_loss = main_loss.new_zeros(())
    weight = 0.0
    if auxiliary_logits is not None:
        auxiliary_logits = F.interpolate(auxiliary_logits, size=masks.shape[-2:],
                                         mode="bilinear", align_corners=False)
        auxiliary_loss = criterion(auxiliary_logits, masks)
        weight = model.config.local_refinement["discrimination_aux_weight"]
    loss = main_loss + weight * auxiliary_loss
    contrastive_loss = None
    if use_contrastive:
        contrastive_loss = contrastive(outputs.decoder_features, images, masks, logits)
        if contrastive_loss is not None:
            loss = loss + contrastive_weight * contrastive_loss
    lovasz_loss = None
    if lovasz_weight > 0:
        lovasz_loss = foreground_lovasz_loss(logits, masks)
        if lovasz_loss is not None:
            loss = loss + lovasz_weight * lovasz_loss
    error_risk_loss, error_coarse_loss = None, None
    error_counts = None
    coarse_logits = getattr(outputs, "coarse_logits", None)
    if coarse_logits is not None:
        settings = model.config.local_refinement
        if coarse_logits.shape[-2:] != masks.shape[-2:]:
            coarse_logits = F.interpolate(coarse_logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
        error_coarse_loss = criterion(coarse_logits, masks)
        error_risk_loss, error_counts = balanced_error_loss(outputs.error_logits, coarse_logits, masks)
        # Preserve total CE coefficient=1; in auxiliary-only mode coarse and
        # final logits coincide, so this mix is exactly the original CE.
        loss = loss + settings["error_coarse_weight"] * (error_coarse_loss - main_loss)
        loss = loss + settings["error_risk_weight"] * error_risk_loss
    return {"loss": loss, "main_loss": main_loss, "auxiliary_loss": auxiliary_loss,
            "auxiliary_weight": weight, "contrastive_loss": contrastive_loss, "lovasz_loss": lovasz_loss,
            "error_risk_loss": error_risk_loss, "error_coarse_loss": error_coarse_loss,
            "error_counts": error_counts}


def train_epoch(model, loader, optimizer, criterion, device: torch.device,
                epoch: int, total_epochs: int, log_every: int, *,
                contrastive: HardAlgaeContrastiveLoss | None = None,
                contrastive_weight: float = 0.0, lovasz_weight: float = 0.0) -> dict:
    logger = logging.getLogger("segformer")
    model.train()
    started = time.monotonic()
    total_loss = 0.0
    total_main_loss, total_auxiliary_loss = 0.0, 0.0
    total_contrastive_loss, contrastive_batches = 0.0, 0
    total_lovasz_loss, lovasz_batches = 0.0, 0
    total_error_risk, total_error_coarse = 0.0, 0.0
    error_counts = np.zeros(2, dtype=np.int64)
    num_images = 0
    bar = tqdm(loader, desc=f"Train {epoch}/{total_epochs}", disable=not sys.stderr.isatty())
    for step, batch in enumerate(bar, start=1):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        losses = training_losses(model, images, masks, criterion, contrastive=contrastive,
                                 contrastive_weight=contrastive_weight, lovasz_weight=lovasz_weight)
        loss = losses["loss"]
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite training loss at epoch {epoch}, batch {step}.")
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        total_main_loss += losses["main_loss"].item() * images.size(0)
        total_auxiliary_loss += losses["auxiliary_loss"].item() * images.size(0)
        if losses["contrastive_loss"] is not None:
            total_contrastive_loss += losses["contrastive_loss"].item() * images.size(0)
            contrastive_batches += 1
        if losses["lovasz_loss"] is not None:
            total_lovasz_loss += losses["lovasz_loss"].item() * images.size(0)
            lovasz_batches += 1
        if losses["error_risk_loss"] is not None:
            total_error_risk += losses["error_risk_loss"].item() * images.size(0)
            total_error_coarse += losses["error_coarse_loss"].item() * images.size(0)
            error_counts += losses["error_counts"].detach().cpu().numpy()
        num_images += images.size(0)
        bar.set_postfix(loss=f"{total_loss / num_images:.5f}", lr=f"{optimizer.param_groups[0]['lr']:.2g}")
        if step == 1 or step % log_every == 0 or step == len(loader):
            logger.info(
                "Train epoch %s/%s batch %s/%s | loss %.6f | encoder lr %.3g | decoder lr %.3g | %.1fs",
                epoch, total_epochs, step, len(loader), total_loss / num_images,
                optimizer.param_groups[0]["lr"], optimizer.param_groups[1]["lr"], time.monotonic() - started,
            )
            if losses["auxiliary_weight"]:
                logger.info("Loss components | main CE %.6f | coarse CE %.6f | coarse weight %.3g",
                            total_main_loss / num_images, total_auxiliary_loss / num_images,
                            losses["auxiliary_weight"])
            if contrastive_weight > 0:
                logger.info("Loss components | main CE %.6f | contrastive %.6f | weight %.3g | active batches %s/%s",
                            total_main_loss / num_images, total_contrastive_loss / num_images,
                            contrastive_weight, contrastive_batches, step)
            if lovasz_weight > 0:
                logger.info("Loss components | main CE %.6f | foreground Lovasz %.6f | weight %.3g | active batches %s/%s",
                            total_main_loss / num_images, total_lovasz_loss / num_images,
                            lovasz_weight, lovasz_batches, step)
            if losses["error_risk_loss"] is not None:
                logger.info("Error refinement | final CE %.6f | initial CE %.6f | risk BCE %.6f | FP/FN targets %s/%s",
                            total_main_loss / num_images, total_error_coarse / num_images,
                            total_error_risk / num_images, int(error_counts[0]), int(error_counts[1]))
    return {"loss": total_loss / num_images, "main_loss": total_main_loss / num_images,
            "auxiliary_loss": total_auxiliary_loss / num_images,
            "contrastive_loss": total_contrastive_loss / num_images,
            "contrastive_batches": contrastive_batches,
            "lovasz_loss": total_lovasz_loss / num_images, "lovasz_batches": lovasz_batches,
            "error_risk_loss": total_error_risk / num_images,
            "error_coarse_loss": total_error_coarse / num_images,
            "error_fp_targets": int(error_counts[0]), "error_fn_targets": int(error_counts[1]),
            "seconds": time.monotonic() - started}


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
    error_diagnostics = None
    bar = tqdm(loader, desc=description, disable=not sys.stderr.isatty())
    for step, batch in enumerate(bar, start=1):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        outputs = forward_with_input_scale(model, images)
        logits = outputs.logits
        if logits.shape[-2:] != masks.shape[-2:]:
            logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
        if getattr(outputs, "coarse_logits", None) is not None:
            if error_diagnostics is None:
                error_diagnostics = ErrorDiagnostics()
            coarse = F.interpolate(outputs.coarse_logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
            risk = F.interpolate(outputs.error_logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
            error_diagnostics.update(coarse, logits, risk, masks)
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
            "error_diagnostics": error_diagnostics.report() if error_diagnostics is not None else None,
            "seconds": time.monotonic() - started}
