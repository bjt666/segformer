"""Predict one four-band TIFF or a folder, with optional ground-truth comparisons."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import tifffile
import torch
from tqdm import tqdm

from segformer_baseline.checkpoint import load_checkpoint, preprocessing_from_checkpoint
from segformer_baseline.dataset import file_sha256, image_files, normalize_image, read_image, read_mask
from segformer_baseline.engine import resized_logits
from segformer_baseline.metrics import METRIC_POLICY, confusion_matrix, metrics_from_confusion
from segformer_baseline.model import model_from_checkpoint
from segformer_baseline.utils import collect_environment, resolve_device, setup_logger, write_csv, write_json
from segformer_baseline.visualization import save_prediction_visuals


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True, help="HxWx4 TIFF or a directory of TIFF images")
    parser.add_argument("--mask-dir", type=Path, help="Optional GT TIFF directory with matching filenames")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save-probabilities", action="store_true", help="Save foreground probability as float32 TIFF")
    parser.add_argument("--log-every", type=int, default=20)
    args = parser.parse_args()
    if args.log_every < 1:
        parser.error("log-every must be positive")
    if args.input.is_file():
        if args.input.suffix.lower() not in {".tif", ".tiff"}:
            parser.error("input must be a four-band TIFF")
        paths = [args.input]
    else:
        paths = image_files(args.input)
    if args.mask_dir:
        missing = [path.name for path in paths if not (args.mask_dir / path.name).is_file()]
        if missing:
            raise FileNotFoundError(f"Missing ground-truth masks: {missing[:5]}")
    checkpoint = load_checkpoint(args.checkpoint)
    channels, mean, std = preprocessing_from_checkpoint(checkpoint)
    device = resolve_device(args.device)
    run_dir = args.checkpoint.parent.parent if args.checkpoint.parent.name == "checkpoints" else args.checkpoint.parent
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    output = args.output_dir or run_dir / "predictions" / stamp
    if output.exists() and any(path.name != "console.log" for path in output.iterdir()):
        raise FileExistsError(f"Prediction output already contains files: {output}")
    logger = setup_logger(output / "predict.log")
    model = model_from_checkpoint(checkpoint).to(device)
    model.eval()
    per_image, input_records = [], []
    confusion = np.zeros((2, 2), dtype=np.int64)
    for index, path in enumerate(tqdm(paths, desc="Predict", disable=not sys.stderr.isatty()), start=1):
        image = read_image(path)
        tensor = normalize_image(image, channels, mean, std).unsqueeze(0).to(device)
        logits = resized_logits(model, tensor, image.shape[:2])
        # Match evaluate.py exactly: argmax assigns class 0 on equal logits.
        prediction = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
        target = read_mask(args.mask_dir / path.name, image.shape[:2]) if args.mask_dir else None
        save_prediction_visuals(path, prediction, output, target=target)
        record = {"name": path.name, "sha256": file_sha256(path), "height": image.shape[0], "width": image.shape[1]}
        if target is not None:
            matrix = confusion_matrix(target, prediction)
            confusion += matrix
            per_image.append({"name": path.name, **metrics_from_confusion(matrix)})
            record["mask_sha256"] = file_sha256(args.mask_dir / path.name)
        input_records.append(record)
        if args.save_probabilities:
            probability = logits.softmax(dim=1)[0, 1].cpu().numpy().astype(np.float32)
            probability_path = output / "probabilities" / path.name
            probability_path.parent.mkdir(parents=True, exist_ok=True)
            tifffile.imwrite(probability_path, probability)
        if index == 1 or index % args.log_every == 0 or index == len(paths):
            logger.info("Predicted %s/%s: %s", index, len(paths), path.name)
    report = {"num_images": len(paths), "checkpoint": str(args.checkpoint.resolve()),
              "checkpoint_sha256": file_sha256(args.checkpoint), "checkpoint_epoch": checkpoint.get("epoch"),
              "input": str(args.input.resolve()), "input_channels": list(channels),
              "input_bands": checkpoint.get("config", {}).get("input_bands", ["NIR", "R", "G"]),
              "mask_encoding": {"background": 0, "target": 255}, "probabilities_saved": args.save_probabilities,
              "display": "NIR,R,G with per-image per-channel 2nd/98th percentile stretch; display only",
              "georeferencing": "Pixel dimensions are preserved; GeoTIFF spatial metadata is not copied.",
              "inputs": input_records}
    if args.mask_dir:
        report.update({"metrics": metrics_from_confusion(confusion), "confusion_matrix": confusion.tolist(),
                       "metric_policy": METRIC_POLICY})
        write_csv(output / "per_image_metrics.csv", per_image)
    write_json(output / "prediction_manifest.json", report)
    write_json(output / "environment.json", collect_environment(Path(__file__).resolve().parent, device))
    logger.info("Prediction masks and visualizations saved to %s", output.resolve())


if __name__ == "__main__":
    main()
