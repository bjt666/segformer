"""Display-only false-color stretching, prediction panels and training curves."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

from .dataset import read_image


def false_color(image: np.ndarray) -> np.ndarray:
    # Display is always NIR,R,G, even if a future experiment selects different input bands.
    rgb = image[..., [3, 2, 1]].astype(np.float32)
    low, high = np.percentile(rgb, [2, 98], axis=(0, 1))
    rgb = np.clip((rgb - low) / np.maximum(high - low, 1e-6), 0, 1)
    return (rgb * 255).astype(np.uint8)


def mask_rgb(mask: np.ndarray) -> np.ndarray:
    return np.repeat((mask.astype(np.uint8) * 255)[..., None], 3, axis=2)


def overlay_mask(rgb: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    overlay = rgb.astype(np.float32).copy()
    foreground = prediction.astype(bool)
    overlay[foreground] = 0.55 * overlay[foreground] + 0.45 * np.array([255, 210, 0])
    return overlay.astype(np.uint8)


def error_rgb(target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    target, prediction = target.astype(bool), prediction.astype(bool)
    rgb = np.zeros((*target.shape, 3), dtype=np.uint8)
    rgb[target & prediction] = [0, 180, 0]
    rgb[~target & prediction] = [255, 0, 0]
    rgb[target & ~prediction] = [0, 100, 255]
    return rgb


def save_panel(path: Path, panels: list[tuple[str, np.ndarray]], footer: str) -> None:
    height, width = panels[0][1].shape[:2]
    canvas = Image.new("RGB", (width * len(panels), height + 72), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (title, array) in enumerate(panels):
        canvas.paste(Image.fromarray(array), (index * width, 28))
        draw.text((index * width + 6, 8), title, fill="black")
    draw.text((6, height + 36), footer, fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def save_prediction_visuals(image_path: Path, prediction: np.ndarray, output_dir: Path,
                           target: np.ndarray | None = None, panels_only: bool = False) -> None:
    rgb = false_color(read_image(image_path))
    overlay = overlay_mask(rgb, prediction)
    filename = image_path.name + ".png"  # Retain .tif/.tiff to avoid same-stem collisions.
    if not panels_only:
        for directory, array in (("masks", prediction.astype(np.uint8) * 255), ("overlays", overlay)):
            path = output_dir / directory / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(array).save(path)
    panels = [("False color (NIR,R,G)", rgb), ("Prediction", mask_rgb(prediction)), ("Overlay", overlay)]
    footer = "Yellow overlay = predicted foreground; display stretch only."
    if target is not None:
        panels = [("False color (NIR,R,G)", rgb), ("Ground truth", mask_rgb(target)),
                  ("Prediction", mask_rgb(prediction)), ("Errors", error_rgb(target, prediction))]
        footer = "Errors: green=TP, red=FP, blue=FN, black=TN. Display stretch only."
    save_panel(output_dir / "comparisons" / filename, panels, footer)


def plot_history(history: list[dict], output_dir: Path) -> None:
    if not history:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = [row["epoch"] for row in history]
    for filename, series, ylabel in (
        ("loss.png", [("train_loss", "Training"), ("val_loss", "Validation")], "Cross entropy"),
        ("metrics.png", [("val_foreground_iou", "Foreground IoU"), ("val_miou", "mIoU"),
                         ("val_dice", "Foreground Dice")], "Validation metric"),
        ("learning_rate.png", [("encoder_lr", "Encoder"), ("decoder_lr", "Decoder")], "Learning rate"),
    ):
        figure, axis = plt.subplots(figsize=(7, 4.5))
        for key, label in series:
            values = [np.nan if row.get(key) is None else row[key] for row in history]
            axis.plot(epochs, values, label=label)
        axis.set_xlabel("Epoch")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
        axis.legend()
        figure.tight_layout()
        figure.savefig(output_dir / filename, dpi=180)
        plt.close(figure)


def plot_confusion(confusion: np.ndarray, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(5, 4))
    axis.imshow(confusion, cmap="Blues")
    for row in range(2):
        for column in range(2):
            axis.text(column, row, str(int(confusion[row, column])), ha="center", va="center",
                      color="white" if confusion[row, column] > confusion.max() / 2 else "black")
    axis.set_xticks([0, 1], ["Background", "Target"])
    axis.set_yticks([0, 1], ["Background", "Target"])
    axis.set_xlabel("Prediction")
    axis.set_ylabel("Ground truth")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)
