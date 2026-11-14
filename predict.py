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

from segformer_baseline.checkpoint import input_bands_from_checkpoint, load_checkpoint, preprocessing_from_checkpoint
from segformer_baseline.dataset import file_sha256, image_files, normalize_image, read_image, read_mask
from segformer_baseline.engine import resized_logits
from segformer_baseline.error_diagnostics import ErrorDiagnostics
from segformer_baseline.input_scale import (
    INPUT_SCALE_POLICY, forward_with_input_scale, logits_on_grid, model_input_scale, scaled_size,
)
from segformer_baseline.metrics import METRIC_POLICY, confusion_matrix, metrics_from_confusion
from segformer_baseline.model import model_architecture, model_from_checkpoint, model_parameter_counts
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
    parser.add_argument("--save-error-maps", action="store_true",
                        help="For error-refinement models, save initial probabilities, FP/FN risk scores and logit corrections")
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
    has_error_refinement = model_architecture(model)["error_refinement"]
    if args.save_error_maps and not has_error_refinement:
        parser.error("--save-error-maps requires an error_refinement checkpoint")
    error_diagnostics = ErrorDiagnostics() if has_error_refinement and args.mask_dir else None
    per_image, input_records = [], []
    confusion = np.zeros((2, 2), dtype=np.int64)
    for index, path in enumerate(tqdm(paths, desc="Predict", disable=not sys.stderr.isatty()), start=1):
        image = read_image(path)
        tensor = normalize_image(image, channels, mean, std).unsqueeze(0).to(device)
        if has_error_refinement:
            outputs = forward_with_input_scale(model, tensor)
            logits = logits_on_grid(outputs.logits, image.shape[:2])
            outputs.coarse_logits = logits_on_grid(outputs.coarse_logits, image.shape[:2])
            outputs.error_logits = logits_on_grid(outputs.error_logits, image.shape[:2])
            outputs.logit_correction = logits_on_grid(outputs.logit_correction, image.shape[:2])
        else:
            logits = resized_logits(model, tensor, image.shape[:2])
        # Match evaluate.py exactly: argmax assigns class 0 on equal logits.
        prediction = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
        target = read_mask(args.mask_dir / path.name, image.shape[:2]) if args.mask_dir else None
        save_prediction_visuals(path, prediction, output, target=target)
        record = {"name": path.name, "sha256": file_sha256(path), "height": image.shape[0], "width": image.shape[1]}
        record["model_input_size"] = list(scaled_size(image.shape[:2], model_input_scale(model)))
        if target is not None:
            matrix = confusion_matrix(target, prediction)
            confusion += matrix
            per_image.append({"name": path.name, **metrics_from_confusion(matrix)})
            record["mask_sha256"] = file_sha256(args.mask_dir / path.name)
            if error_diagnostics is not None:
                error_diagnostics.update(outputs.coarse_logits, logits, outputs.error_logits,
                                         torch.from_numpy(target).unsqueeze(0).to(device))
        input_records.append(record)
        if args.save_probabilities:
            probability = logits.softmax(dim=1)[0, 1].cpu().numpy().astype(np.float32)
            probability_path = output / "probabilities" / path.name
            probability_path.parent.mkdir(parents=True, exist_ok=True)
            tifffile.imwrite(probability_path, probability)
        if args.save_error_maps:
            maps = {"coarse_probabilities": outputs.coarse_logits.softmax(dim=1)[0, 1],
                    "fp_risk_scores": outputs.error_logits.sigmoid()[0, 0],
                    "fn_risk_scores": outputs.error_logits.sigmoid()[0, 1],
                    "logit_corrections": outputs.logit_correction[0, 0]}
            for category, values in maps.items():
                destination = output / category / path.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                tifffile.imwrite(destination, values.cpu().numpy().astype(np.float32))
        if index == 1 or index % args.log_every == 0 or index == len(paths):
            logger.info("Predicted %s/%s: %s", index, len(paths), path.name)
    report = {"num_images": len(paths), "checkpoint": str(args.checkpoint.resolve()),
              "checkpoint_sha256": file_sha256(args.checkpoint), "checkpoint_epoch": checkpoint.get("epoch"),
              "input": str(args.input.resolve()), "input_channels": list(channels),
              "input_bands": input_bands_from_checkpoint(checkpoint),
              "input_scale": model_input_scale(model), "input_scale_policy": INPUT_SCALE_POLICY,
              "architecture": model_architecture(model), "parameter_counts": model_parameter_counts(model),
              "mask_encoding": {"background": 0, "target": 255}, "probabilities_saved": args.save_probabilities,
              "error_maps_saved": args.save_error_maps,
              "error_map_policy": "FP scores apply only where initial argmax is foreground; FN scores only where "
                  "it is background. Raw sigmoid risk scores are saved and are not calibrated probabilities. "
                  "Correction is added to the foreground logit; negative suppresses, positive increases foreground.",
              "display": "NIR,R,G with per-image per-channel 2nd/98th percentile stretch; display only",
              "georeferencing": "Pixel dimensions are preserved; GeoTIFF spatial metadata is not copied.",
              "inputs": input_records}
    if args.mask_dir:
        report.update({"metrics": metrics_from_confusion(confusion), "confusion_matrix": confusion.tolist(),
                       "metric_policy": METRIC_POLICY})
        write_csv(output / "per_image_metrics.csv", per_image)
        if error_diagnostics is not None:
            write_json(output / "error_diagnostics.json", {
                "checkpoint_sha256": report["checkpoint_sha256"], "checkpoint_epoch": report["checkpoint_epoch"],
                **error_diagnostics.report(),
            })
    write_json(output / "prediction_manifest.json", report)
    write_json(output / "environment.json", collect_environment(Path(__file__).resolve().parent, device))
    logger.info("Prediction masks and visualizations saved to %s", output.resolve())


if __name__ == "__main__":
    main()
