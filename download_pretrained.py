"""Save ImageNet encoder weights to a reusable local system-disk directory."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from segformer_baseline.hybrid import RESNET50_URL


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("/root/models"))
    parser.add_argument("--cnn-only", action="store_true", help="Keep using the existing cached MiT encoder")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    target = args.output_dir / "resnet50_imagenet1k_v2.pth"
    if not target.exists():
        temporary = target.with_suffix(".download")
        try:
            torch.hub.download_url_to_file(RESNET50_URL, str(temporary), hash_prefix="11ad3fa6")
            temporary.replace(target)
        finally:
            if temporary.exists():
                temporary.unlink()
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    if not digest.startswith("11ad3fa6"):
        raise ValueError(f"Existing ResNet file has the wrong SHA256: {target}; use a new output directory.")
    if not args.cnn_only:
        from transformers import SegformerForImageClassification
        mit = args.output_dir / "mit-b0"
        SegformerForImageClassification.from_pretrained("nvidia/mit-b0").save_pretrained(mit)
        print(f"MiT encoder saved: {mit.resolve()}")
    record = {"resnet50_source": RESNET50_URL, "path": str(target.resolve()), "sha256": digest}
    (args.output_dir / "resnet50_source.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
