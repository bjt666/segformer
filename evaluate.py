"""Evaluate a checkpoint on validation/test data and export reproducible metrics."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from torch import nn
from torch.utils.data import DataLoader

from segformer_baseline.checkpoint import load_checkpoint, preprocessing_from_checkpoint
from segformer_baseline.dataset import PairedTiffDataset, build_manifest, file_sha256
from segformer_baseline.engine import evaluate_model
from segformer_baseline.metrics import METRIC_POLICY
from segformer_baseline.model import model_from_checkpoint
from segformer_baseline.utils import collect_environment, resolve_device, setup_logger, write_csv, write_json
from segformer_baseline.visualization import plot_confusion, save_prediction_visuals


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("dataset"))
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--visualization-samples", type=int, default=8)
    args = parser.parse_args()
    if args.batch_size < 1 or args.num_workers < 0 or args.log_every < 1 or args.visualization_samples < 0:
        parser.error("batch-size/log-every must be positive; num-workers/visualization-samples must be nonnegative")
    checkpoint = load_checkpoint(args.checkpoint)
    channels, mean, std = preprocessing_from_checkpoint(checkpoint)
    device = resolve_device(args.device)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    run_dir = args.checkpoint.parent.parent if args.checkpoint.parent.name == "checkpoints" else args.checkpoint.parent
    output = args.output_dir or run_dir / "evaluation" / f"{args.split}-{stamp}"
    if output.exists() and any(path.name != "console.log" for path in output.iterdir()):
        raise FileExistsError(f"Evaluation output already contains files: {output}")
    logger = setup_logger(output / "evaluate.log")
    dataset = PairedTiffDataset(args.data_root, args.split, mean, std, channels)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=device.type == "cuda")
    model = model_from_checkpoint(checkpoint).to(device)
    result = evaluate_model(model, loader, device, nn.CrossEntropyLoss(),
                            description=f"Evaluate {args.split}", log_every=args.log_every,
                            sample_limit=args.visualization_samples)
    report = {"split": args.split, "num_images": result["num_images"],
              "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": file_sha256(args.checkpoint),
              "checkpoint_epoch": checkpoint.get("epoch"), "input_channels": list(channels),
              "input_bands": checkpoint.get("config", {}).get("input_bands", ["NIR", "R", "G"]),
              "loss": result["loss"], **result["metrics"],
              "confusion_matrix": result["confusion"].tolist(), "metric_policy": METRIC_POLICY}
    write_json(output / "metrics.json", report)
    write_csv(output / "per_image_metrics.csv", result["per_image"])
    write_csv(output / "confusion_matrix.csv", [
        {"ground_truth": "background", "predicted_background": int(result["confusion"][0, 0]),
         "predicted_target": int(result["confusion"][0, 1])},
        {"ground_truth": "target", "predicted_background": int(result["confusion"][1, 0]),
         "predicted_target": int(result["confusion"][1, 1])},
    ])
    plot_confusion(result["confusion"], output / "confusion_matrix.png")
    for sample in result["samples"]:
        save_prediction_visuals(Path(sample["image_path"]), sample["prediction"],
                                output / "visualizations", target=sample["target"])
    logger.info("Hashing evaluation files for the data manifest.")
    write_json(output / "dataset_manifest.json", build_manifest(args.data_root, (args.split,)))
    write_json(output / "environment.json", collect_environment(Path(__file__).resolve().parent, device))
    logger.info("Results saved to %s", output.resolve())
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
