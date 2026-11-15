#!/usr/bin/env bash
# Run the two new ablations. Existing single-scale outputs are not touched.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Optionally: PRETRAINED_MODEL=/root/models/mit-b0 bash scripts/run_shared_scale.sh
PRETRAINED_ARGS=()
if [[ -n "${PRETRAINED_MODEL:-}" ]]; then
  PRETRAINED_ARGS=(--pretrained-model "$PRETRAINED_MODEL")
fi

# Check BOTH before starting any long training. Settings match batch=8, native=256.
for MODE in plain interaction; do
  python check_model.py --config "configs/segformer_b0_decoder_freqfusion_lovasz_shared_${MODE}.yaml"
done

for MODE in plain interaction; do
  CONFIG="configs/segformer_b0_decoder_freqfusion_lovasz_shared_${MODE}.yaml"
  RUN="outputs/gf16m_decoder_freqfusion_lovasz_shared_${MODE}_s42"
  if [[ -f "$RUN/analysis_bundle.zip" ]]; then
    echo "Completed analysis archive already exists: $RUN/analysis_bundle.zip (skipping)"
    continue
  fi
  if [[ -f "$RUN/checkpoints/last.pt" ]]; then
    python train.py --resume "$RUN/checkpoints/last.pt"
  else
    python train.py --config "$CONFIG" --output-dir "$RUN" "${PRETRAINED_ARGS[@]}"
  fi
  # Ctrl-C can leave a valid last.pt while train.py exits successfully. Never
  # evaluate/package an unfinished run as if it were a completed experiment.
  python - "$RUN" <<'PY'
import json, sys
from pathlib import Path
run = Path(sys.argv[1])
summary = json.loads((run / 'summary.json').read_text())
config = __import__('yaml').safe_load((run / 'config.yaml').read_text())
if summary['status'] != 'completed' or summary['last_completed_epoch'] != config['epochs']:
    raise SystemExit('Training incomplete. Re-run this script to resume from last.pt.')
PY
  # Reuse completed exports; CLI guards prevent overwriting partial directories.
  for SPLIT in val test; do
    if [[ ! -f "$RUN/evaluation/$SPLIT/metrics.json" || ! -f "$RUN/evaluation/$SPLIT/per_image_metrics.csv" \
          || ! -f "$RUN/evaluation/$SPLIT/dataset_manifest.json" ]]; then
      python evaluate.py --data-root dataset --checkpoint "$RUN/checkpoints/best.pt" \
        --split "$SPLIT" --output-dir "$RUN/evaluation/$SPLIT"
    fi
  done
  if [[ ! -f "$RUN/predictions/val/prediction_manifest.json" ]]; then
    python predict.py --checkpoint "$RUN/checkpoints/best.pt" \
      --input dataset/images/val --mask-dir dataset/masks/val --save-probabilities \
      --output-dir "$RUN/predictions/val"
  fi
  python export_analysis.py --run-dir "$RUN"
done

echo 'Download the analysis_bundle.zip from each of the two shared-scale output folders.'
