"""Inspect trained refinement modules with temporary validation-time interventions.

The same checkpoint is used for every intervention. No training, checkpoint
editing, or pretrained-weight download is performed. These interventions are
diagnostic comparisons, not replacements for separately trained ablations.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from segformer_baseline.checkpoint import input_bands_from_checkpoint, load_checkpoint, preprocessing_from_checkpoint
from segformer_baseline.dataset import PairedTiffDataset, build_manifest, file_sha256
from segformer_baseline.engine import resized_logits
from segformer_baseline.metrics import METRIC_POLICY, confusion_matrix, metrics_from_confusion
from segformer_baseline.model import model_architecture, model_from_checkpoint, model_parameter_counts
from segformer_baseline.refinement import RefinedDecodeHead
from segformer_baseline.utils import collect_environment, resolve_device, setup_logger, write_csv, write_json


GATE_HISTOGRAM_BINS = 65536
RATE_METRICS = (
    "foreground_iou", "background_iou", "miou", "dice", "foreground_f1",
    "precision", "recall", "pixel_accuracy",
)


def image_rms(value: torch.Tensor) -> list[float]:
    value = value.detach().float()
    if not torch.isfinite(value).all():
        raise FloatingPointError("Nonfinite refinement features encountered during diagnosis.")
    return value.square().flatten(1).mean(dim=1).sqrt().cpu().tolist()


def relative_rms(numerator: float, feature_rms: float) -> float | None:
    """Zero reference features have no defined relative residual magnitude."""
    return numerator / feature_rms if feature_rms > 0.0 else None


class GateDistribution:
    """Bounded-memory pooled gate statistics; no feature maps are retained."""

    def __init__(self, initial_probability: float) -> None:
        self.initial_probability = initial_probability
        self.count = 0
        self.total = self.total_square = self.total_absolute_initial_difference = 0.0
        self.minimum, self.maximum = 1.0, 0.0
        self.histogram = np.zeros(GATE_HISTOGRAM_BINS, dtype=np.int64)

    def update(self, values: torch.Tensor) -> None:
        flat = values.detach().float().reshape(-1)
        if not torch.isfinite(flat).all():
            raise FloatingPointError("Nonfinite gate values encountered during diagnosis.")
        self.count += flat.numel()
        self.total += flat.double().sum().item()
        self.total_square += flat.double().square().sum().item()
        self.total_absolute_initial_difference += (flat.double() - self.initial_probability).abs().sum().item()
        self.minimum = min(self.minimum, flat.min().item())
        self.maximum = max(self.maximum, flat.max().item())
        indices = (flat * GATE_HISTOGRAM_BINS).long().clamp_(0, GATE_HISTOGRAM_BINS - 1)
        self.histogram += torch.bincount(indices, minlength=GATE_HISTOGRAM_BINS).cpu().numpy()

    def report(self) -> dict:
        if not self.count:
            raise RuntimeError("Expected gate observations, but none were collected.")
        mean = self.total / self.count
        cumulative = np.cumsum(self.histogram)
        quantiles = {}
        for label, fraction in (("q05", 0.05), ("q50", 0.50), ("q95", 0.95)):
            rank = max(1, math.ceil(fraction * self.count))
            index = int(np.searchsorted(cumulative, rank, side="left"))
            quantiles[label] = (index + 0.5) / GATE_HISTOGRAM_BINS
        return {
            "count": self.count, "mean": mean,
            "pooled_std": math.sqrt(max(0.0, self.total_square / self.count - mean * mean)),
            "minimum": self.minimum, "maximum": self.maximum,
            "initial_probability": self.initial_probability,
            "mean_minus_initial_probability": mean - self.initial_probability,
            "mean_absolute_difference_from_initial_probability": self.total_absolute_initial_difference / self.count,
            **quantiles,
        }


class RefinementObserver:
    """Observe natural activations, or replace only selected stage outputs."""

    def __init__(self, head: RefinedDecodeHead, architecture: dict) -> None:
        self.head, self.architecture = head, architecture
        self.handles = []
        self.mode = "normal"
        self.names: list[str] = []
        self.cache: dict[str, dict] = {}
        self.rows: list[dict] = []
        self.gate_distributions: dict[str, list[GateDistribution]] = {}

    def install(self) -> None:
        for stage, module in self.head.refinements.items():
            for index, branch in enumerate(module.branches):
                self.handles.append(branch.register_forward_hook(self._branch_hook(stage, index)))
            if module.module_b:
                self.gate_distributions[stage] = [
                    GateDistribution(self.architecture["gate_init_probability"]) for _ in module.branches
                ]
                self.handles.append(module.gate.register_forward_hook(self._gate_hook(stage)))
            self.handles.append(module.register_forward_hook(self._stage_hook(stage)))

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.cache.clear()

    def start(self, mode: str, names: list[str]) -> None:
        self.mode, self.names = mode, list(names)
        self.cache.clear()

    def _branch_hook(self, stage: str, index: int):
        def hook(module, inputs, output):
            self.cache.setdefault(stage, {}).setdefault("residuals", {})[index] = output.detach()
        return hook

    def _gate_hook(self, stage: str):
        def hook(module, inputs, output):
            if self.mode == "normal":
                self.cache.setdefault(stage, {})["gates"] = output.detach().float().sigmoid()
        return hook

    def _stage_hook(self, stage: str):
        def hook(module, inputs, output):
            features = inputs[0]
            if self.mode == "no_residual":
                return features
            observed = self.cache[stage]
            residuals = [observed["residuals"][index] for index in range(len(module.branches))]
            mean_residual = torch.stack(residuals, dim=0).mean(dim=0)
            if self.mode == "uniform_gates":
                return features + mean_residual
            if self.mode != "normal":
                raise ValueError(f"Unknown intervention: {self.mode}")
            self._record(stage, features, residuals, mean_residual, output, observed.get("gates"))
            return None
        return hook

    def _record(self, stage: str, features: torch.Tensor, residuals: list[torch.Tensor],
                mean_residual: torch.Tensor, output: torch.Tensor, gates: torch.Tensor | None) -> None:
        feature_rms = image_rms(features)
        mean_rms = image_rms(mean_residual)
        delta_rms = image_rms(output - features)
        branch_rms = [image_rms(residual) for residual in residuals]
        gate_columns = {}
        if gates is not None:
            for index, distribution in enumerate(self.gate_distributions[stage]):
                values = gates[:, index]
                distribution.update(values)
                flat = values.flatten(1)
                gate_columns[f"gate_{index}_mean"] = flat.mean(dim=1).cpu().tolist()
                gate_columns[f"gate_{index}_spatial_std"] = flat.std(dim=1, unbiased=False).cpu().tolist()
                gate_columns[f"gate_{index}_mean_absolute_difference_from_initial"] = (
                    flat - self.architecture["gate_init_probability"]
                ).abs().mean(dim=1).cpu().tolist()
        for image_index, name in enumerate(self.names):
            row = {
                "name": name, "stage": int(stage), "feature_rms": feature_rms[image_index],
                "mean_candidate_rms": mean_rms[image_index], "actual_delta_rms": delta_rms[image_index],
                "mean_candidate_to_feature_rms_ratio": relative_rms(mean_rms[image_index], feature_rms[image_index]),
                "actual_delta_to_feature_rms_ratio": relative_rms(delta_rms[image_index], feature_rms[image_index]),
            }
            for index, values in enumerate(branch_rms):
                row[f"branch_{index}_kernel"] = self.architecture["local_kernels"][index]
                row[f"branch_{index}_rms"] = values[image_index]
                row[f"branch_{index}_to_feature_rms_ratio"] = relative_rms(values[image_index], feature_rms[image_index])
            row.update({key: values[image_index] for key, values in gate_columns.items()})
            self.rows.append(row)

    def stage_reports(self) -> list[dict]:
        reports = []
        for stage, module in self.head.refinements.items():
            rows = [row for row in self.rows if row["stage"] == int(stage)]
            summary = {"stage": int(stage), "num_images": len(rows)}
            for key in rows[0]:
                if key in {"name", "stage"} or key.endswith("_kernel"):
                    continue
                values = np.asarray([row[key] for row in rows if row[key] is not None], dtype=np.float64)
                summary[f"{key}_defined_images"] = int(values.size)
                summary[f"{key}_undefined_images"] = len(rows) - int(values.size)
                statistics = {"mean": None, "std": None, "min": None, "max": None,
                              "q05": None, "q50": None, "q95": None}
                if values.size:
                    statistics.update(mean=float(values.mean()), std=float(values.std()),
                                      min=float(values.min()), max=float(values.max()))
                    statistics.update(zip(("q05", "q50", "q95"),
                                          (float(value) for value in np.quantile(values, [0.05, 0.50, 0.95]))))
                summary.update({f"{key}_{suffix}": value for suffix, value in statistics.items()})
            report = {"stage": int(stage), "per_image_summary": summary,
                      "branches": [{"index": index, "kernel": kernel}
                                   for index, kernel in enumerate(self.architecture["local_kernels"])]}
            if module.module_b:
                final_layer = module.gate[-1]
                weight, bias = final_layer.weight.detach().double(), final_layer.bias.detach().double()
                initial_bias = math.log(self.architecture["gate_init_probability"] /
                                        (1.0 - self.architecture["gate_init_probability"]))
                report["gate_final_convolution"] = {
                    "weight_l2": weight.norm().item(), "weight_max_absolute": weight.abs().max().item(),
                    "initial_weight_value": 0.0, "initial_bias_value": initial_bias,
                    "bias": bias.cpu().tolist(),
                    "bias_max_absolute_difference_from_initial": (bias - initial_bias).abs().max().item(),
                }
                report["branches"] = [
                    {"index": index, "kernel": self.architecture["local_kernels"][index],
                     "gate_distribution": distribution.report()}
                    for index, distribution in enumerate(self.gate_distributions[stage])
                ]
            reports.append(report)
        return reports


class InterventionTotals:
    def __init__(self) -> None:
        self.confusion = np.zeros((2, 2), dtype=np.int64)
        self.cross_entropy_sum = self.probability_difference_sum = self.union_probability_difference_sum = 0.0
        self.pixels = self.changed_pixels = self.union_pixels = self.union_changed_pixels = self.images = 0

    def update(self, logits: torch.Tensor, target: torch.Tensor, reference_probability: torch.Tensor,
               reference_prediction: torch.Tensor, names: list[str]) -> list[dict]:
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Nonfinite prediction logits encountered during diagnosis.")
        probability = logits.float().softmax(dim=1)[:, 1]
        prediction = logits.argmax(dim=1)
        difference = (probability - reference_probability).abs()
        changed = prediction != reference_prediction
        union = target.bool() | prediction.bool() | reference_prediction.bool()
        # Match evaluate_model: batch-mean CE weighted by batch image count.
        self.cross_entropy_sum += F.cross_entropy(logits.float(), target, reduction="mean").item() * target.shape[0]
        self.probability_difference_sum += difference.double().sum().item()
        self.union_probability_difference_sum += difference[union].double().sum().item()
        self.pixels += target.numel()
        self.changed_pixels += changed.sum().item()
        self.union_pixels += union.sum().item()
        self.union_changed_pixels += (changed & union).sum().item()
        self.images += target.shape[0]
        rows = []
        targets, predictions = target.cpu().numpy(), prediction.cpu().numpy()
        for index, name in enumerate(names):
            matrix = confusion_matrix(targets[index], predictions[index])
            self.confusion += matrix
            count = int(union[index].sum().item())
            rows.append({
                "name": name, **metrics_from_confusion(matrix),
                "foreground_probability_mad": difference[index].double().mean().item(),
                "prediction_changed_fraction": changed[index].float().mean().item(),
                "foreground_union_pixels": count,
                "foreground_union_probability_mad": difference[index][union[index]].double().mean().item() if count else None,
                "foreground_union_changed_fraction": (changed[index] & union[index]).sum().item() / count if count else None,
            })
        return rows

    def report(self, mode: str) -> dict:
        return {
            "mode": mode, "num_images": self.images, "num_pixels": self.pixels,
            "cross_entropy": self.cross_entropy_sum / self.images,
            **metrics_from_confusion(self.confusion), "confusion_matrix": self.confusion.tolist(),
            "foreground_probability_mad": self.probability_difference_sum / self.pixels,
            "prediction_changed_pixels": self.changed_pixels,
            "prediction_changed_fraction": self.changed_pixels / self.pixels,
            "foreground_union_pixels": self.union_pixels,
            "foreground_union_probability_mad": self.union_probability_difference_sum / self.union_pixels if self.union_pixels else None,
            "foreground_union_changed_fraction": self.union_changed_pixels / self.union_pixels if self.union_pixels else None,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("dataset"))
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log-every", type=int, default=20)
    args = parser.parse_args()
    if args.batch_size < 1 or args.num_workers < 0 or args.log_every < 1:
        parser.error("batch-size/log-every must be positive; num-workers must be nonnegative")

    checkpoint = load_checkpoint(args.checkpoint)
    channels, mean, std = preprocessing_from_checkpoint(checkpoint)
    device = resolve_device(args.device)
    model = model_from_checkpoint(checkpoint).to(device).eval()
    architecture = model_architecture(model)
    if not isinstance(model.decode_head, RefinedDecodeHead) or not architecture["module_a"]:
        parser.error("This checkpoint has no module A. Diagnose a trained +A or +A+B checkpoint instead.")
    modes = ["normal", "no_residual"]
    if architecture["module_b"]:
        modes.append("uniform_gates")
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    run_dir = args.checkpoint.parent.parent if args.checkpoint.parent.name == "checkpoints" else args.checkpoint.parent
    output = args.output_dir or run_dir / "diagnostics" / f"{args.split}-{stamp}"
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Diagnostic output must be a new or empty directory: {output}")
    dataset = PairedTiffDataset(args.data_root, args.split, mean, std, channels, augment=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                        pin_memory=device.type == "cuda")
    logger = setup_logger(output / "diagnose.log")
    logger.info("Diagnosing %s on %s (%s images); modes=%s", args.checkpoint, args.split, len(dataset), modes)
    totals = {mode: InterventionTotals() for mode in modes}
    observer = RefinementObserver(model.decode_head, architecture)
    per_image_rows = []
    started = time.monotonic()
    try:
        observer.install()
        with torch.inference_mode():
            bar = tqdm(loader, desc=f"Diagnose {args.split}", disable=not sys.stderr.isatty())
            for step, batch in enumerate(bar, start=1):
                images = batch["image"].to(device, non_blocking=True)
                target = batch["mask"].to(device, non_blocking=True)
                names = list(batch["name"])
                observer.start("normal", names)
                normal_logits = resized_logits(model, images, target.shape[-2:])
                normal_probability = normal_logits.float().softmax(dim=1)[:, 1]
                normal_prediction = normal_logits.argmax(dim=1)
                for mode in modes:
                    if mode == "normal":
                        logits = normal_logits
                    else:
                        observer.start(mode, names)
                        logits = resized_logits(model, images, target.shape[-2:])
                    rows = totals[mode].update(logits, target, normal_probability, normal_prediction, names)
                    per_image_rows.extend({"mode": mode, **row} for row in rows)
                if step == 1 or step % args.log_every == 0 or step == len(loader):
                    logger.info("Diagnose batch %s/%s | images per mode=%s | %.1fs", step, len(loader),
                                {mode: totals[mode].images for mode in modes}, time.monotonic() - started)
        stage_reports = observer.stage_reports()
    finally:
        observer.remove()

    interventions = [totals[mode].report(mode) for mode in modes]
    reference = interventions[0]
    for row in interventions:
        row["delta_cross_entropy_vs_normal"] = row["cross_entropy"] - reference["cross_entropy"]
        for key in RATE_METRICS:
            row[f"delta_{key}_pp_vs_normal"] = (
                100.0 * (row[key] - reference[key]) if row[key] is not None and reference[key] is not None else None
            )
    logger.info("Hashing diagnostic files for the data manifest.")
    manifest = build_manifest(args.data_root, (args.split,))
    environment = collect_environment(Path(__file__).resolve().parent, device)
    report = {
        "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": file_sha256(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"), "split": args.split, "num_images": len(dataset),
        "data_root": str(args.data_root.resolve()), "architecture": architecture,
        "parameter_counts": model_parameter_counts(model), "training_config": checkpoint.get("config"),
        "checkpoint_training_environment": checkpoint.get("environment"),
        "checkpoint_training_dataset_fingerprint": checkpoint.get("dataset_fingerprint"),
        "input_channels": list(channels), "input_bands": input_bands_from_checkpoint(checkpoint),
        "normalization_mean": mean.tolist(), "normalization_std": std.tolist(),
        "dataset_fingerprint": manifest["fingerprint"], "dataset_manifest_file": "dataset_manifest.json",
        "environment_file": "environment.json", "metric_policy": METRIC_POLICY,
        "interventions": interventions, "stages": stage_reports,
        "diagnostic_policy": {
            "normal": "Unmodified forward pass of this trained checkpoint.",
            "no_residual": "At every selected refinement stage replace the output F' with its input F; encoder and trained original decoder are unchanged.",
            "uniform_gates": "For +A+B only, replace each selected stage output by F + mean(R_k), with K arbitrary; other trained parameters stay unchanged.",
            "comparison": "Every intervention is compared batch-by-batch with the natural forward pass of the SAME checkpoint in eval/inference mode, without augmentation or autocast.",
            "cross_entropy": "Same aggregation as evaluate_model: batch-mean CE times batch image count, divided by total image count; no class weights.",
            "probability_mad": "Mean absolute difference of foreground softmax probabilities, on the 0-to-1 probability scale.",
            "foreground_union": "Pairwise union of GT foreground, normal predicted foreground, and this intervention's predicted foreground; empty union ratios are null.",
            "rms": "Per image, over all channel and spatial elements. Ratios divide by the measured feature RMS; when that RMS is zero the ratio is null, excluded from summaries and counted as undefined. There is no automatic ineffective/effective threshold.",
            "stage_summary": "Unweighted mean/std/min/max and exact q05/q50/q95 of defined per-image statistics; std uses population normalization. Defined and undefined image counts are reported separately.",
            "gates": "Statistics describe sigmoid gates (independent across branches), not logits. Spatial std is computed within each image, then summarized across images.",
            "gate_quantiles": {
                "method": "Pooled inverse empirical CDF (ceil(q*N) order statistic), approximated by midpoint of its uniform [0,1] histogram bin.",
                "bins": GATE_HISTOGRAM_BINS, "absolute_error_bound": 1.0 / GATE_HISTOGRAM_BINS,
                "other_gate_moments": "Means/std/min/max and initialization deviations are accumulated from actual gate values, without histogram approximation.",
            },
            "interpretation": [
                "A small residual, nearly uniform gates, or changed parameters alone cannot establish usefulness or failure; inspect interventions and validation metrics together.",
                "Bypassing residuals is not the independently trained baseline. Uniform gating is not the independently trained +A model.",
                "These inference-time changes can introduce a distribution shift. They show checkpoint dependence, not a unique cause or the outcome of retraining.",
                "A gate can learn a useful global reweighting even when its spatial std is small. Final-convolution weights were initially zero; bias was logit(initial_probability).",
                "Independent sigmoid gates do not sum to one. Replacing them with 1/K changes both branch reweighting and total residual amplitude; this control alone cannot isolate spatial selection or prove use of deep semantic context.",
                "No optimizer update or checkpoint write is performed. Only normal forward passes contribute to feature/gate statistics.",
            ],
        },
        "elapsed_seconds": time.monotonic() - started,
    }
    write_json(output / "diagnostic_report.json", report)
    write_csv(output / "interventions.csv", [
        {key: value for key, value in row.items() if key != "confusion_matrix"} for row in interventions
    ])
    write_csv(output / "stage_statistics.csv", [stage["per_image_summary"] for stage in stage_reports])
    write_csv(output / "per_image_stage_statistics.csv", observer.rows)
    write_csv(output / "per_image_interventions.csv", per_image_rows)
    write_json(output / "dataset_manifest.json", manifest)
    write_json(output / "environment.json", environment)
    logger.info("Results saved to %s", output.resolve())
    print(json.dumps({"output_dir": str(output.resolve()), "interventions": interventions}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
