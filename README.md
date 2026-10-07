# SegFormer baseline for GF false-color semantic segmentation

This repository contains a reproducible binary-segmentation baseline for the paired four-band TIFF dataset in `dataset/`. The loader selects GF's NIR, red, and green bands in that order to make a three-channel false-color input; `train.py` and `evaluate.py` provide the training and held-out test entry points.

## Dataset layout

```text
dataset/
  images/{train,val,test}/*.tif   # 256x256x4, float16
  masks/{train,val,test}/*.tif    # 256x256, uint8; 0=background, 255=target
```

The dataset is intentionally excluded from Git. Keep it on local/AutoDL persistent storage and pass its path with `--data-root`.

## Environment

Use Python 3.10 or 3.11. Create an environment and install a CUDA-enabled PyTorch build matching the server's driver/GPU first, using the command from the official PyTorch installation selector. Then install this project's packages:

```bash
pip install -r requirements.txt
```

The baseline uses Hugging Face Transformers' SegFormer implementation. By default, it loads the **ImageNet-1K-pretrained MiT-B0 encoder** (`nvidia/mit-b0`) and initializes a new binary segmentation decoder/head. TIFFs are assumed to store GF bands as **Blue, Green, Red, NIR**; the loader selects `[NIR, Red, Green]` as the model input. For the pretrained first patch projection, the NIR filter is initialized from the mean RGB filter, while red and green map to their matching ImageNet filters. Per-band mean and standard deviation are computed from the selected training bands only, saved with the checkpoint, and reused for validation/test. The training script performs flips and 90-degree rotations only, which preserve band and mask alignment.

## Training

```bash
python train.py --data-root ./dataset --output-dir ./outputs/segformer-b0 --epochs 100 --batch-size 8
```

To train without downloading pretrained weights, add `--no-pretrained`. To select another ImageNet-pretrained SegFormer image-classification checkpoint, pass `--pretrained-model <model-id-or-local-path>`.

The training script saves the best validation checkpoint and JSON records of the run configuration and metrics. It reports foreground IoU, Dice, precision, recall, and pixel accuracy. The test split is not used during training; evaluate it separately after selecting the model:

```bash
python evaluate.py --data-root ./dataset --checkpoint ./outputs/segformer-b0/best.pt
```

## Reproducibility

Record the Git commit, checkpoint source, random seed, Python/PyTorch/Transformers versions, GPU model, and the exact dataset version for each experiment. Do not commit datasets, checkpoints, or generated outputs.

## Data notes

The observed training split contains 700 paired images, with 87 validation and 89 test pairs. TIFF metadata records only the array shape, so the GF B-G-R-NIR storage order is an explicit assumption based on the user's description; confirm that the export did not reorder bands. For input-band ablations, compare the primary NIR-R-G false-color input with true-color R-G-B and the full four-band input, keeping all other training settings fixed. NDVI can be evaluated as an additional input only after documenting the reflectance/radiometric preprocessing. The loader expects masks whose target pixels are any nonzero value (the current masks use 255).
