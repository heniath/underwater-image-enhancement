# Physics-Guided Underwater Image Enhancement

U-Net trained with physics-guided input channels (UDCP transmission map + background light) on the EUVP benchmark.

---

## Environment Setup (Conda)

### 1. Create and activate the conda environment

```bash
conda create -n uwir python=3.10 -y
conda activate uwir
```

### 2. Install PyTorch (CUDA 12.1)

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

### 3. Install remaining dependencies

```bash
pip install -r requirements.txt
```

> **Note**: PyTorch is already installed via conda, so pip will skip it.
> If pip tries to reinstall a CPU-only build, use
> `pip install -r requirements.txt --no-deps torch torchvision torchaudio` instead.

### 4. Verify the installation

```bash
python - <<'EOF'
import torch, torchvision, kornia, cv2, skimage, thop
print("torch     :", torch.__version__, "| CUDA:", torch.cuda.is_available())
print("torchvision:", torchvision.__version__)
print("kornia    :", kornia.__version__)
print("opencv    :", cv2.__version__)
print("scikit-img:", skimage.__version__)
print("thop      :", thop.__version__)
EOF
```

---

## Quick-start (one-liner after first-time setup)

```bash
conda activate uwir && python net_test.py
```

---

## Dataset Setup

Download datasets and place them under `./datasets/` with the following structure:

```
datasets/
  EUVP/
    Paired/
      underwater_imagenet/
        trainA/          ← degraded inputs
        trainB/          ← clean references
        testA/
        testB/
      underwater_dark/
        trainA/ trainB/ testA/ testB/
      underwater_scenes/
        trainA/ trainB/ testA/ testB/
  UIEB/
    raw-890/             ← degraded inputs
    reference-890/       ← reference images
  UFO120/
    train_val/
      lrd/               ← low-res / degraded
      hr/                ← high-quality reference
    test/
      lrd/
      hr/
  U45/                   ← 45 no-reference images (flat folder)
```

---

## Training

```bash
conda activate uwir

# Full 5-channel model (RGB + transmission map + background light)
python train.py --model unet_5ch --dataset euvp

# RGB-only baseline
python train.py --model unet_3ch --dataset euvp

# Combined EUVP + UIEB training
python train.py --model unet_5ch --dataset euvp+uieb

# Custom hyper-parameters
python train.py \
    --model unet_5ch \
    --dataset euvp \
    --batchSize 8 \
    --nEpochs 200 \
    --lr 1e-4 \
    --cos_restart True \
    --warmup_epochs 5 \
    --early_stop_patience 20
```

Key arguments (see `data/options.py` for the full list):

| Argument | Default | Description |
|---|---|---|
| `--model` | `unet_5ch` | Model variant (`unet_3ch`, `unet_5ch`, …) |
| `--dataset` | `euvp` | Training set (`euvp`, `uieb`, `ufo120`, `euvp+uieb`) |
| `--batchSize` | `16` | Mini-batch size |
| `--nEpochs` | `200` | Total epochs |
| `--lr` | `1e-4` | Initial learning rate |
| `--cos_restart` | `True` | Cosine annealing with restarts |
| `--warmup_epochs` | `3` | Linear LR warm-up epochs |
| `--early_stop_patience` | `20` | Early stopping patience (epochs) |
| `--checkpoint_dir` | `./checkpoints/` | Where to save `.pth` files |

Checkpoints are saved to `./checkpoints/` and the training history JSON to `./results/`.

---

## Evaluation

```bash
python eval.py
```

---

## Model Profiling

```bash
python net_test.py
```

Prints inference time, parameter count (M), and FLOPs (G) for a `256×256` input.

---

## Ablation Variants

| Variant | Input channels | Description |
|---|---|---|
| `unet_3ch` | 3 | RGB-only baseline |
| `unet_4ch_t` | 4 | RGB + transmission map t(x) |
| `unet_4ch_b` | 4 | RGB + background light B |
| **`unet_5ch`** | **5** | **RGB + t(x) + B ← proposed** |

---

## Project Structure

```
underwater-image-enhancement/
├── data/
│   ├── UWIRdataset.py   — UIEB, EUVP, UFO-120, U45 dataset classes
│   ├── data.py          — Dataset factory functions
│   ├── eval_sets.py     — Padded / simple eval loaders
│   ├── options.py       — All training arguments (argparse)
│   ├── scheduler.py     — GradualWarmup, CosineRestartLR schedulers
│   └── util.py          — is_image_file, load_img helpers
├── net/
│   ├── unet.py          — UNet5ch model (3- or 5-channel input)
│   └── physics.py       — UDCP transmission map + background light
├── loss/
│   └── losses.py        — CompositeLoss (L1 + VGG Perceptual + SSIM)
├── measure_underwater.py — PSNR, SSIM, CIEDE2000, UCIQE, UIQM metrics
├── train.py             — Main training loop
├── eval.py              — Test-set evaluation
├── net_test.py          — Model profiling (time / params / FLOPs)
├── requirements.txt     — pip dependencies
└── README.md
```

---

## Results

| Variant | Val PSNR | Test PSNR | Val SSIM | Test SSIM |
|---|---|---|---|---|
| `unet_3ch` (RGB only) | — | — | — | — |
| `unet_5ch` (proposed) | — | — | — | — |

*Fill in after training.*
