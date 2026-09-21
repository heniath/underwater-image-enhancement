# Physics-Informed Underwater Image Restoration (UWIR)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](tests/)

This repository provides a modular, high-performance framework for **Physics-Informed Underwater Image Restoration (UWIR)**. It combines physical optical imaging priors (transmission and backscatter estimation via UDCP) with modern deep learning and lightweight restoration architectures, including **SGMA-Net** (Mamba-Attention), **FA-Net / FA-Net+**, **NAFNet**, **U-Net**, and **UW-LYT**.

---

## 🌟 Key Features

- **Physics-Guided Input Formulations**: Flexible model contracts supporting 3, 4, and 5-channel inputs:
  - `3ch`: Standard RGB input.
  - `4ch_t`: RGB + Transmission map ($t(x)$).
  - `4ch_b`: RGB + Background / Backscatter light map ($B_\infty$).
  - `5ch`: RGB + both Transmission and Backscatter physics priors.
- **Comprehensive Model Zoo**:
  - **SGMA-Net** (Lightweight Mamba-Attention Network): Combines selective state-space sequence modeling with spatial attention.
  - **FA-Net / FA-Net+**: Frequency and feature attention networks for detail sharpening and color correction.
  - **NAFNet-Tiny**: Non-linear activation-free architecture for efficient restoration.
  - **FGDPA / FGDPA-Slim** (Frequency-Guided Dual-Path Attention, ICME 2026): Ultra-lightweight re-parameterizable architecture (4.23K parameters, 800+ FPS).
  - **LiteEnhanceNet / LSNet / MobileIE**: Edge-optimized architectures for mobile and embedded deployment.
  - **Customized U-Net & UW-LYT**: Paper baseline architectures.
- **Standardized Datasets**: Built-in loaders for **EUVP** (Underwater Dark, Imagenet, Scenes) and **UIEB** (890 paired real-world images).
- **Comprehensive Metrics Suite**:
  - Full-Reference: PSNR, SSIM, CIEDE2000.
  - No-Reference: UCIQE, UIQM.
  - Efficiency: Params (M), FLOPs (G), Latency (ms), and FPS.
- **Tiled Inference**: Seamless processing of arbitrarily high-resolution images without GPU out-of-memory errors.

---

## 📂 Repository Structure

```text
underwater-image-enhancement/
├── src/uwir/                       # Core package
│   ├── cli/                        # Command-line tools (train, evaluate, profile)
│   ├── data/                       # Dataset loaders (EUVP, UIEB, factory)
│   ├── models/                     # Model zoo (SGMA-Net, FA-Net+, NAFNet, U-Net, etc.)
│   ├── physics/                    # Optical priors (UDCP transmission & backscatter)
│   ├── training/                   # Learning rate schedulers & optimizers
│   ├── losses.py                   # CompositeLoss (L1, Perceptual, SSIM, Gradient)
│   ├── metrics.py                  # PSNR, SSIM, CIEDE2000, UCIQE, UIQM, tiled inference
│   └── config.py                   # Training & benchmark configuration
│
├── scripts/                        # Experiment & utility scripts
│   ├── inference.py                # Direct inference on images/folders
│   ├── setup_wsl_mamba.sh          # Environment setup for Linux/WSL
│   ├── diagnostics/                # Physics map visualization & verification
│   ├── experiments/                # Multi-seed ablation studies & bash runners
│   └── visualization/              # Publication-ready qualitative & quantitative plots
│
├── tests/                          # 12 unit and integration test suites
├── docs/                           # Published results & literature reviews
├── datasets/                       # Dataset placeholder with layout guide
├── checkpoints/                    # Saved weights (.pth) placeholder
├── results/                        # Output evaluation logs and images
├── logs/                           # Training run logs
├── pyproject.toml                  # Build system & package metadata
└── requirements.txt                # Python package dependencies
```

---

## 🚀 Getting Started

### 1. Prerequisites & Installation

Python **3.10** or newer is required.

```bash
# Clone the repository
git clone https://github.com/heniath/underwater-image-enhancement.git
cd underwater-image-enhancement

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate       # On Linux/WSL
# or .\.venv\Scripts\activate   # On Windows

# Install PyTorch with CUDA (e.g. CUDA 12.1)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install requirements and package in editable mode
pip install -e .
pip install -e ".[dev,profile,visualization]"
```

### 2. Verify Installation with Tests

Run the test suite to ensure all dependencies and model registries are functioning properly (tests run immediately on CPU using synthetic dummy tensors):

```bash
pytest
```

---

## 📊 Dataset Preparation

Organize your datasets in the `datasets/` directory as follows:

```text
datasets/
├── EUVP/
│   ├── Paired/
│   │   ├── underwater_imagenet/trainA/ (degraded)  trainB/ (clean)
│   │   ├── underwater_dark/trainA/      trainB/
│   │   └── underwater_scenes/trainA/    trainB/
│   └── test_samples/Inp/ (degraded)  GTr/ (ground truth)
│
└── UIEB/
    ├── raw-890/             # 890 degraded real-world images
    └── reference-890/       # Corresponding ground truth references
```

### UIEB Train/Test Split (800 Train / 90 Test)

To follow the standard research convention dividing UIEB into **800 training pairs** and **90 test pairs**, run the deterministic split script:

```bash
# Split UIEB-890 into 800 train and 90 test (reproducible seed=42)
python scripts/split_uieb.py --data_dir ./datasets/UIEB --seed 42

# Optional: use symlinks instead of copying files (requires admin privileges on Windows)
python scripts/split_uieb.py --data_dir ./datasets/UIEB --seed 42 --symlink
```

After running the script, the UIEB dataset directory will be organized as:

```text
datasets/UIEB/
├── raw-890/             # 890 degraded real-world images (untouched)
├── reference-890/       # 890 ground truth references (untouched)
├── train/
│   ├── input/           # 800 degraded images for training
│   └── reference/       # 800 corresponding GT images
├── test/
│   ├── input/           # 90 degraded images for evaluation
│   └── reference/       # 90 corresponding GT images
└── split_manifest.txt   # Manifest recording the exact stems for reproducibility
```

> **Note**: Both `uwir-train` and `uwir-evaluate` will automatically detect and prioritize `train/` and `test/` subsets if present, seamlessly falling back to `raw-890/` if unsplit.

Refer to [`datasets/README.md`](datasets/README.md) for official download links and extraction details.

---

## 🖼️ Inference (Enhance Your Own Images)

Enhance a single image or a folder of images using a trained checkpoint:

```bash
# Enhance a folder of images
python scripts/inference.py \
  --checkpoint checkpoints/best_model.pth \
  --model sgmanet_5ch \
  --input_dir path/to/raw_images \
  --output_dir results/enhanced_images \
  --device cuda

# Enhance a single image
python scripts/inference.py \
  --checkpoint checkpoints/best_model.pth \
  --model sgmanet_5ch \
  --input_image path/to/underwater.jpg \
  --output_dir results/enhanced_images
```

> **Tip**: For high-resolution images (e.g. 4K), `--tile_size 512 --tile_overlap 64` is enabled by default to prevent CUDA OOM while seamlessly stitching tiles.

---

## ⚡ Model Zoo & Profiling

Inspect model parameters, FLOPs, and runtime speed:

```bash
# List all registered models
uwir-profile --list

# Profile SGMA-Net on GPU
uwir-profile sgmanet_5ch --device cuda --img_size 256

# Profile FA-Net+ on CPU
uwir-profile fanetplus_5ch --device cpu --img_size 256
```

### Model Variants Overview

| Model | Variants | Description |
|---|---|---|
| **SGMA-Net** | `sgmanet_3ch`, `sgmanet_4ch_t`, `sgmanet_4ch_b`, `sgmanet_5ch` | Lightweight Mamba State-Space + Attention |
| **FA-Net+** | `fanetplus_3ch`, `fanetplus_5ch` | Enhanced Frequency Attention Network |
| **FA-Net** | `fanet_3ch`, `fanet_5ch` | Frequency Attention Network baseline |
| **NAFNet-Tiny** | `nafnettiny_3ch` | Nonlinear Activation Free Network |
| **LiteEnhanceNet** | `liteenhancenet_3ch` | High-throughput lightweight architecture |
| **LSNet** | `lsnet_3ch` | Lightweight Spectral Network |
| **MobileIE** | `mobileie_3ch` | Mobile-optimized inverted bottleneck design |
| **U-Net** | `unet_3ch`, `unet_4ch_t`, `unet_4ch_b`, `unet_5ch` | Paper baseline study architecture |
| **UW-LYT** | `uwlyt_3ch`, `uwlyt_5ch`, `uwlyttiny_3ch`, `uwlyttiny_5ch` | Multi-scale lightweight model |

---

## 🏋️ Training & Evaluation

### Train a Model

```bash
uwir-train \
  --model sgmanet_5ch \
  --dataset euvp \
  --data_train_euvp ./datasets/EUVP \
  --nEpochs 100 \
  --batchSize 4 \
  --cropSize 256 \
  --lr 2e-4 \
  --cos_restart true \
  --L1_weight 1.0 \
  --perceptual_weight 1.0 \
  --SSIM_weight 0.1 \
  --seed 0
```

### Evaluate Checkpoints

Evaluate all models in a checkpoint directory against full-reference and no-reference benchmarks:

```bash
uwir-evaluate \
  --eval_benchmark euvp \
  --data_train_euvp ./datasets/EUVP \
  --checkpoint_dir ./checkpoints \
  --val_folder ./results/euvp
```

### Automated Ablation Runs

Run multi-seed ablation experiments across variants:

```bash
# EUVP ablation
python -m scripts.experiments.ablation_euvp \
  --variants sgmanet_3ch sgmanet_4ch_t sgmanet_4ch_b sgmanet_5ch

# UIEB ablation
python -m scripts.experiments.ablation_uieb \
  --variants sgmanet_3ch sgmanet_4ch_t sgmanet_4ch_b sgmanet_5ch
```

Or run all automated tournament experiments:

```bash
bash scripts/experiments/run_sgmanet_all.sh
```

---

## 📓 Notebooks

- [`uwlytms_kaggle.ipynb`](uwlytms_kaggle.ipynb): Ready-to-run interactive training and evaluation pipeline on Kaggle GPU.
- [`uwlytms_local.ipynb`](uwlytms_local.ipynb): Local prototyping and visualization notebook.

---

## 📄 Citation & License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

```bibtex
@article{uwir2026,
  title={Physics-Informed Underwater Image Restoration},
  author={Heniath and Contributors},
  journal={EIDT},
  year={2026}
}
```
