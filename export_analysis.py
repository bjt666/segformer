"""Package only metrics, provenance and validation probabilities for analysis."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


REQUIRED_FILES = (
    "metrics.csv", "config.yaml", "environment.json", "dataset_manifest.json", "preprocessing.json",
    "evaluation/val/metrics.json", "evaluation/val/per_image_metrics.csv",
    "evaluation/test/metrics.json", "evaluation/test/per_image_metrics.csv",
    "evaluation/test/dataset_manifest.json", "predictions/val/prediction_manifest.json",
)


def package_analysis(run_dir: Path, output: Path | None = None) -> dict:
    run_dir = run_dir.resolve()
    paths = [run_dir / name for name in REQUIRED_FILES]
    missing = [str(path.relative_to(run_dir)) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required analysis files are missing: {missing}. Finish val/test evaluation and val prediction first.")
    reports = [json.loads((run_dir / name).read_text(encoding="utf-8")) for name in (
        "evaluation/val/metrics.json", "evaluation/test/metrics.json", "predictions/val/prediction_manifest.json",
    )]
    if reports[0]["split"] != "val" or reports[1]["split"] != "test":
        raise ValueError("Evaluation folders do not contain the expected val/test splits.")
    for split, report in zip(("val", "test"), reports):
        with (run_dir / f"evaluation/{split}/per_image_metrics.csv").open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if (len(rows) != report["num_images"]
                or any(sum(int(row[key]) for row in rows) != report[key] for key in ("tn", "fp", "fn", "tp"))):
            raise ValueError(f"{split} per-image metrics do not sum to the reported dataset confusion counts.")
    if len({report["checkpoint_sha256"] for report in reports}) != 1:
        raise ValueError("Validation, test and prediction reports were generated from different checkpoints.")
    if len({report.get("input_scale", 1.0) for report in reports}) != 1:
        raise ValueError("Validation, test and prediction input scales disagree.")
    from segformer_baseline.model import architecture_from_config
    architectures = [architecture_from_config(report.get("architecture", {})) for report in reports]
    if any(architecture != architectures[0] for architecture in architectures[1:]):
        raise ValueError("Validation, test and prediction architectures disagree.")
    preprocessing = json.loads((run_dir / "preprocessing.json").read_text(encoding="utf-8"))
    if preprocessing.get("input_scale", 1.0) != reports[0].get("input_scale", 1.0):
        raise ValueError("Training preprocessing and evaluation input scales disagree.")
    mode = reports[0].get("architecture", {}).get("shared_scale", "off")
    if preprocessing.get("shared_scale", "off") != mode:
        raise ValueError("Training preprocessing and evaluation shared-scale modes disagree.")
    best_checkpoint = run_dir / "checkpoints/best.pt"
    if best_checkpoint.is_file():
        digest = hashlib.sha256()
        with best_checkpoint.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != reports[0]["checkpoint_sha256"]:
            raise ValueError("Evaluation reports do not describe the current best.pt; re-evaluate the saved best model.")
    prediction = reports[-1]
    expected = {record["name"] for record in prediction["inputs"]}
    probabilities = sorted((run_dir / "predictions/val/probabilities").glob("*.tif*"))
    if (not prediction.get("probabilities_saved") or len(probabilities) != prediction["num_images"]
            or {path.name for path in probabilities} != expected
            or prediction["num_images"] != reports[0]["num_images"]):
        raise ValueError("Validation probability TIFFs are missing or do not match the validation report.")
    prediction_hashes = {record["name"]: record["sha256"] for record in prediction["inputs"]}
    manifest = json.loads((run_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
    validation_hashes = {Path(record["path"]).name: record["sha256"] for record in manifest["files"]
                         if record["path"].startswith("images/val/")}
    if prediction_hashes != validation_hashes:
        raise ValueError("Predicted images do not match the training run's validation image hashes.")
    # This manifest is checked but not duplicated in the ZIP: its validation
    # records are already included in the root train/val manifest.
    val_manifest_path = run_dir / "evaluation/val/dataset_manifest.json"
    if not val_manifest_path.is_file():
        raise FileNotFoundError("Validation evaluation dataset_manifest.json is missing.")
    val_manifest = json.loads(val_manifest_path.read_text(encoding="utf-8"))
    validation_records = {record["path"]: record["sha256"] for record in manifest["files"]
                          if record["path"].startswith(("images/val/", "masks/val/"))}
    if validation_records != {record["path"]: record["sha256"] for record in val_manifest["files"]}:
        raise ValueError("Validation evaluation images/masks differ from the training run's validation files.")
    paths.extend(probabilities)
    destination = output.resolve() if output is not None else run_dir / "analysis_bundle.zip"
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation protects a previously downloaded/generated archive.
    with ZipFile(destination, mode="x", compression=ZIP_DEFLATED) as archive:
        for path in paths:
            archive.write(path, arcname=(Path(run_dir.name) / path.relative_to(run_dir)).as_posix())
    return {"archive": str(destination), "files": len(paths), "validation_probability_images": len(probabilities),
            "input_scale": reports[0].get("input_scale", 1.0), "bytes": destination.stat().st_size}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="Optional new ZIP path; existing archives are never overwritten")
    args = parser.parse_args()
    print(json.dumps(package_analysis(args.run_dir, args.output), indent=2))


if __name__ == "__main__":
    main()
