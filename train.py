"""Train or resume the GF SegFormer baseline at completed epoch boundaries."""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from segformer_baseline.checkpoint import (
    SCHEMA_VERSION, atomic_torch_save, load_checkpoint, make_checkpoint,
    preprocessing_from_checkpoint, restore_rng,
)
from segformer_baseline.config import DEFAULTS, default_config, read_config, save_config, validate_config
from segformer_baseline.dataset import PairedTiffDataset, build_manifest, compute_train_stats
from segformer_baseline.engine import evaluate_model, train_epoch
from segformer_baseline.model import build_model, model_from_checkpoint
from segformer_baseline.utils import (
    collect_environment, resolve_device, save_pip_freeze, seed_worker, set_seed,
    setup_logger, utc_now, write_history, write_json,
)
from segformer_baseline.visualization import plot_history, save_prediction_visuals


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="YAML configuration; omitted fields use baseline defaults")
    parser.add_argument("--resume", type=Path, help="New-format last.pt or best.pt with full training state")
    for field in ("data_root", "output_dir"):
        parser.add_argument("--" + field.replace("_", "-"), type=Path)
    for field in ("epochs", "batch_size", "num_workers", "seed", "log_every", "visualization_samples"):
        parser.add_argument("--" + field.replace("_", "-"), type=int)
    for field in ("encoder_lr", "decoder_lr", "weight_decay"):
        parser.add_argument("--" + field.replace("_", "-"), type=float)
    parser.add_argument("--device")
    parser.add_argument("--pretrained-model")
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false", default=None)
    return parser.parse_args()


def prepare_config(args: argparse.Namespace, checkpoint: dict | None) -> dict:
    overrides = read_config(args.config)
    overrides.update({key: str(value) if isinstance(value, Path) else value
                      for key, value in vars(args).items()
                      if key in DEFAULTS and value is not None})
    if checkpoint is None:
        config = default_config()
    else:
        required = {"optimizer", "scheduler", "rng_state", "history", "dataset_fingerprint", "best_iou", "best_epoch"}
        if checkpoint.get("schema_version") != SCHEMA_VERSION or not required.issubset(checkpoint):
            raise ValueError(
                "This checkpoint lacks full training state. Old best.pt files support evaluation/prediction, "
                "but cannot resume training. Use a new-format checkpoints/last.pt."
            )
        config = checkpoint["config"].copy()
        if args.resume.parent.name == "checkpoints":
            # The full run directory can be moved to another server without retaining old absolute paths.
            config["output_dir"] = str(args.resume.resolve().parent.parent)
        mutable = {"data_root", "output_dir", "device", "num_workers", "log_every", "visualization_samples"}
        for key, value in overrides.items():
            if key == "output_dir" and value is None:
                continue
            if key not in mutable and value != config[key]:
                raise ValueError(
                    f"Cannot change {key} during resume ({config[key]!r} -> {value!r}). "
                    "Keep the original schedule/settings; start a new experiment for changes."
                )
        overrides = {key: value for key, value in overrides.items() if not (key == "output_dir" and value is None)}
    config.update(overrides)
    validate_config(config)
    if config["output_dir"] is None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        config["output_dir"] = f"outputs/segformer-b0-seed{config['seed']}-{stamp}"
    config["data_root"] = str(Path(config["data_root"]).resolve())
    config["output_dir"] = str(Path(config["output_dir"]).resolve())
    return config


def main() -> None:
    args = parse_args()
    checkpoint = load_checkpoint(args.resume) if args.resume else None
    config = prepare_config(args, checkpoint)
    device = resolve_device(config["device"])
    output = Path(config["output_dir"])
    source_output = (args.resume.resolve().parent.parent if args.resume.parent.name == "checkpoints"
                     else Path(checkpoint["config"]["output_dir"]).resolve()) if checkpoint else None
    if output.exists() and (checkpoint is None or output != source_output):
        occupied = [path for path in output.iterdir() if path.name not in {"console.log", "nohup.out"}]
        if occupied:
            raise FileExistsError(f"Experiment directory already contains files: {output}. Use --resume or a new directory.")
    output.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(output / "train.log")
    logger.info("Experiment directory: %s", output)
    logger.info("Input bands %s from source indices %s; device %s", config["input_bands"], config["input_channels"], device)
    if checkpoint and config["num_workers"] != checkpoint["config"]["num_workers"]:
        logger.warning("num_workers changed. Augmentation RNG sequences may differ from the original run.")
    set_seed(config["seed"])
    root = Path(config["data_root"])
    logger.info("Hashing train/val images and masks for the dataset manifest (test split is not used).")
    manifest = build_manifest(root, ("train", "val"))
    if checkpoint:
        if manifest["fingerprint"] != checkpoint["dataset_fingerprint"]:
            raise ValueError("Training/validation files changed since the saved checkpoint. Resume refused.")
        channels, mean, std = preprocessing_from_checkpoint(checkpoint)
    else:
        channels = tuple(config["input_channels"])
        logger.info("Computing normalization from selected training bands only.")
        mean, std = compute_train_stats(root, channels)
    logger.info("Normalization mean=%s std=%s", mean.tolist(), std.tolist())
    train_set = PairedTiffDataset(root, "train", mean, std, channels, augment=True)
    val_set = PairedTiffDataset(root, "val", mean, std, channels)
    logger.info("Dataset: %s train / %s validation images", len(train_set), len(val_set))
    generators = {"train": torch.Generator().manual_seed(config["seed"]),
                  "val": torch.Generator().manual_seed(config["seed"] + 1)}
    common = {"batch_size": config["batch_size"], "num_workers": config["num_workers"],
              "pin_memory": device.type == "cuda", "worker_init_fn": seed_worker,
              # Recreate workers each epoch to recover their seeds from saved loader RNG states.
              "persistent_workers": False}
    train_loader = DataLoader(train_set, shuffle=True, generator=generators["train"], **common)
    val_loader = DataLoader(val_set, shuffle=False, generator=generators["val"], **common)
    model = (model_from_checkpoint(checkpoint) if checkpoint else build_model(config)).to(device)
    optimizer = torch.optim.AdamW([
        {"params": model.segformer.parameters(), "lr": config["encoder_lr"]},
        {"params": model.decode_head.parameters(), "lr": config["decoder_lr"]},
    ], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
    criterion = nn.CrossEntropyLoss()
    history = checkpoint["history"].copy() if checkpoint else []
    best_iou = checkpoint["best_iou"] if checkpoint else -1.0
    best_epoch = checkpoint["best_epoch"] if checkpoint else 0
    start_epoch = checkpoint["epoch"] + 1 if checkpoint else 1
    if checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        logger.info("Resuming %s at epoch %s/%s; best foreground IoU %.6f at epoch %s",
                    args.resume, start_epoch, config["epochs"], best_iou, best_epoch)
        # Checkpoint history is authoritative, including when an older checkpoint is deliberately resumed.
        history = [row for row in history if row["epoch"] <= checkpoint["epoch"]]
        write_history(output, history)
    session_environment = collect_environment(Path(__file__).resolve().parent, device)
    environment = checkpoint["environment"] if checkpoint else session_environment
    write_json(output / "environment.json", environment)
    sessions_path = output / "sessions.json"
    sessions = json.loads(sessions_path.read_text(encoding="utf-8")) if sessions_path.exists() else []
    sessions.append({"resume_checkpoint": str(args.resume.resolve()) if args.resume else None,
                     "start_epoch": start_epoch, "config": config, "environment": session_environment})
    write_json(sessions_path, sessions)
    save_config(config, output / "config.yaml")
    write_json(output / "run_config.json", config)
    write_json(output / "preprocessing.json", {"source_order": ["B", "G", "R", "NIR"],
               "input_channels": list(channels), "input_bands": config["input_bands"],
               "mean": mean.tolist(), "std": std.tolist(), "statistics_split": "train"})
    write_json(output / "dataset_manifest.json", manifest)
    lock_name = f"requirements-lock-resume-{datetime.now():%Y%m%d-%H%M%S-%f}.txt" if checkpoint else "requirements-lock.txt"
    save_pip_freeze(output / lock_name)
    checkpoints = output / "checkpoints"
    # Validate/restore the historical best, including interruption between saving last.pt and best.pt.
    if checkpoint:
        original_best = source_output / "checkpoints" / "best.pt"
        if checkpoint["epoch"] == best_epoch:
            atomic_torch_save(checkpoint, checkpoints / "best.pt")
        elif original_best.exists():
            best_payload = load_checkpoint(original_best)
            if (best_payload.get("epoch") != best_epoch
                    or best_payload.get("best_iou") != best_iou
                    or best_payload.get("dataset_fingerprint") != manifest["fingerprint"]):
                raise ValueError("The original best.pt does not match the resumed run.")
            if output != source_output:
                atomic_torch_save(best_payload, checkpoints / "best.pt")
        else:
            raise FileNotFoundError(
                "The historical best.pt is missing. Copy the full experiment directory (both last.pt and best.pt) "
                "to preserve the best validation model."
            )
    if checkpoint:
        restore_rng(checkpoint["rng_state"], generators)
    for epoch in range(start_epoch, config["epochs"] + 1):
        epoch_started = time.monotonic()
        encoder_lr, decoder_lr = (group["lr"] for group in optimizer.param_groups)
        train_result = train_epoch(model, train_loader, optimizer, criterion, device,
                                   epoch, config["epochs"], config["log_every"])
        val_result = evaluate_model(model, val_loader, device, criterion,
                                    description=f"Validation {epoch}/{config['epochs']}",
                                    log_every=config["log_every"], collect_per_image=False,
                                    sample_limit=config["visualization_samples"])
        val_iou = val_result["metrics"]["foreground_iou"]
        if val_iou is None:
            raise ValueError("Validation foreground IoU is undefined; check that validation masks contain the target class.")
        improved = val_iou > best_iou
        if improved:
            best_iou, best_epoch = val_iou, epoch
        scheduler.step()
        row = {"epoch": epoch, "train_loss": train_result["loss"], "val_loss": val_result["loss"],
               **{f"val_{key}": value for key, value in val_result["metrics"].items()},
               "encoder_lr": encoder_lr, "decoder_lr": decoder_lr,
               "next_encoder_lr": optimizer.param_groups[0]["lr"],
               "next_decoder_lr": optimizer.param_groups[1]["lr"],
               "train_seconds": train_result["seconds"], "val_seconds": val_result["seconds"],
               "epoch_seconds": time.monotonic() - epoch_started,
               "best_epoch": best_epoch, "best_foreground_iou": best_iou}
        history.append(row)
        payload = make_checkpoint(model, optimizer, scheduler, epoch, best_iou, best_epoch,
                                  config, mean, std, history, manifest, generators, environment)
        atomic_torch_save(payload, checkpoints / "last.pt")
        if improved:
            atomic_torch_save(payload, checkpoints / "best.pt")
        write_history(output, history)
        plot_history(history, output / "curves")
        # Fixed sorted validation samples make previews comparable across epochs.
        for sample in val_result["samples"]:
            save_prediction_visuals(Path(sample["image_path"]), sample["prediction"],
                                    output / "visualizations" / f"epoch_{epoch:04d}",
                                    target=sample["target"], panels_only=True)
        write_json(output / "summary.json", {"status": "training", "last_completed_epoch": epoch,
                   "best_epoch": best_epoch, "best_foreground_iou": best_iou, "updated_at": utc_now()})
        logger.info("Epoch %s complete | train loss %.6f | val loss %.6f | fg IoU %.6f | Dice %s | best epoch %s",
                    epoch, row["train_loss"], row["val_loss"], val_iou, row["val_dice"], best_epoch)
    plot_history(history, output / "curves")
    write_json(output / "summary.json", {"status": "completed", "last_completed_epoch": config["epochs"],
               "best_epoch": best_epoch, "best_foreground_iou": best_iou, "updated_at": utc_now()})
    logger.info("Training completed. Best validation foreground IoU %.6f at epoch %s; checkpoint %s",
                best_iou, best_epoch, checkpoints / "best.pt")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.getLogger("segformer").warning(
            "Interrupted. Resume from checkpoints/last.pt if at least one epoch completed; repeat the unfinished epoch."
        )
        raise SystemExit(130)
    except Exception:
        logging.getLogger("segformer").exception("Training failed. Resolve the error before resuming the last completed epoch.")
        raise
