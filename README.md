# Physics-Guided Underwater Image Enhancement

PyTorch training code for underwater image enhancement with optional
physics-guided input channels. The model variants support RGB-only input,
RGB plus a transmission map, RGB plus a background-light map, or RGB plus both.

Supported physics prior extractors:

- `udcp`: original lightweight underwater dark-channel baseline.
- `gdcp`: Ucolor/GDCP-style transmission and background-light extraction.
- `gupdm`: GUPDM-style transmission-guided physical prior.

See [PHYSICS_PRIORS.md](PHYSICS_PRIORS.md) for a detailed explanation of the
three extraction methods.

---

## Environment Setup

### 1. Create and activate the conda environment

```bash
conda create -n uwir python=3.10 -y
conda activate uwir
```

### 2. Install PyTorch with CUDA 12.1

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

### 3. Install remaining dependencies

```bash
pip install -r requirements.txt
```

If PyTorch is already installed by conda, pip should skip it. If pip attempts
to reinstall a CPU-only build, install the remaining packages manually or use
your environment's preferred PyTorch install command.

### 4. Verify the installation

```bash
python - <<'EOF'
import torch, torchvision, kornia, cv2, skimage, thop
print("torch      :", torch.__version__, "| CUDA:", torch.cuda.is_available())
print("torchvision:", torchvision.__version__)
print("kornia     :", kornia.__version__)
print("opencv     :", cv2.__version__)
print("scikit-img :", skimage.__version__)
print("thop       :", thop.__version__)
EOF
```

---

## Quick Start

```bash
conda activate uwir
python net_test.py
```

---

## Dataset Setup

Place datasets under `./datasets/` or pass custom paths through CLI arguments.

Expected layout:

```text
datasets/
  EUVP/
    Paired/
      underwater_imagenet/
        trainA/          <- degraded inputs
        trainB/          <- clean references
      underwater_dark/
        trainA/
        trainB/
      underwater_scenes/
        trainA/
        trainB/
  UIEB/
    raw-890/             <- degraded inputs
    reference-890/       <- reference images
  UFO120/
    train_val/
      lrd/               <- low-quality inputs
      hr/                <- high-quality references
    test/
      lrd/
      hr/
  U45/                   <- no-reference images, flat folder
```

For Kaggle or other hosted environments, pass the mounted dataset path, for
example:

```bash
python train.py \
  --model unet_5ch \
  --dataset euvp \
  --data_train_euvp /kaggle/input/datasets/pamuduranasinghe/euvp-dataset/EUVP
```

---

## Training

### Basic runs

```bash
# Full 5-channel model: RGB + t(x) + B
python train.py --model unet_5ch --dataset euvp

# RGB-only baseline
python train.py --model unet_3ch --dataset euvp

# RGB + transmission map only
python train.py --model unet_4ch_t --dataset euvp

# RGB + background-light map only
python train.py --model unet_4ch_b --dataset euvp
```

### Select a physics prior extractor

```bash
# Original UDCP baseline
python train.py --model unet_5ch --dataset euvp --prior_method udcp

# Ucolor/GDCP-style prior
python train.py --model unet_5ch --dataset euvp --prior_method gdcp

# GUPDM-style prior
python train.py --model unet_5ch --dataset euvp --prior_method gupdm
```

### Example Kaggle command

```bash
python train.py \
  --model unet_5ch \
  --dataset euvp \
  --data_train_euvp /kaggle/input/datasets/pamuduranasinghe/euvp-dataset/EUVP \
  --batchSize 16 \
  --nEpochs 50 \
  --threads 2 \
  --num_gpus 2 \
  --prior_method gdcp \
  --run_name unet5ch_gdcp_euvp
```

### Custom hyperparameters

```bash
python train.py \
  --model unet_5ch \
  --dataset euvp \
  --prior_method udcp \
  --batchSize 8 \
  --nEpochs 200 \
  --lr 1e-4 \
  --cos_restart True \
  --warmup_epochs 5 \
  --early_stop_patience 20
```

Key arguments:

| Argument | Default | Description |
|---|---:|---|
| `--model` | `unet_5ch` | Model variant: `3ch`, `4ch_t`, `4ch_b`, or `5ch` across supported backbones. |
| `--dataset` | `euvp` | Training set: `euvp`, `uieb`, `ufo120`, or `euvp+uieb`. |
| `--prior_method` | `udcp` | Physics extractor: `udcp`, `gdcp`, or `gupdm`. |
| `--batchSize` | `16` | Mini-batch size. |
| `--nEpochs` | `200` | Total training epochs. |
| `--lr` | `1e-4` | Initial learning rate. |
| `--threads` | `4` | DataLoader workers. Physics extraction runs in the loading path. |
| `--checkpoint_dir` | `./checkpoints/` | Directory for saved `.pth` checkpoints. |
| `--run_name` | empty | Optional run prefix for checkpoint folder naming. |

Checkpoints are saved under `./checkpoints/`. Training history is saved beside
the checkpoint files.

---

## Evaluation

```bash
python eval.py \
  --checkpoint_dir ./checkpoints/ \
  --data_train_euvp ./datasets/EUVP \
  --prior_method udcp
```

Use the same `--prior_method` that was used for the checkpoint you want to
evaluate.

---

## Model Profiling

```bash
python net_test.py
```

Prints inference time, parameter count, and FLOPs for a `256x256` input.

---

## Ablation Variants

| Variant | Input channels | Description |
|---|---:|---|
| `*_3ch` | 3 | RGB-only baseline. |
| `*_4ch_t` | 4 | RGB + transmission/prior map `t(x)`. |
| `*_4ch_b` | 4 | RGB + scalar background-light map `B`. |
| `*_5ch` | 5 | RGB + `t(x)` + `B`. |

Supported backbones include:

- `unet`
- `resnet`
- `mobilenet`
- `mambavision`
- `mambaunet`

Examples:

```bash
python train.py --model resnet_5ch --dataset euvp --prior_method gdcp
python train.py --model mobilenet_4ch_t --dataset euvp --prior_method gupdm
python train.py --model mambavision_3ch --dataset euvp
```

---

## Physics Prior Files

```text
net/physics.py        - UDCP baseline extractor
net/physics_gdcp.py   - Ucolor/GDCP extractor
net/physics_gupdm.py  - GUPDM-style extractor
PHYSICS_PRIORS.md     - Detailed method explanation
```

The training code selects the extractor through `--prior_method` and keeps the
same model input layout for all methods.

---

## Project Structure

```text
underwater-image-enhancement/
  data/
    UWIRdataset.py       - UIEB, EUVP, UFO-120, U45 dataset classes
    data.py              - Dataset factory functions
    eval_sets.py         - Evaluation dataset loaders
    options.py           - CLI arguments
    scheduler.py         - Warmup and cosine scheduler helpers
    util.py              - Image loading helpers
  loss/
    losses.py            - Composite loss
  net/
    physics.py           - UDCP baseline
    physics_gdcp.py      - Ucolor/GDCP prior
    physics_gupdm.py     - GUPDM-style prior
    registry.py          - Model factory and variant parsing
    unet.py              - U-Net model
    resnet_unet.py       - ResNet U-Net model
    mobilenet_unet.py    - MobileNet U-Net model
    mambavision_unet.py  - MambaVision U-Net model
    mamba_unet.py        - Mamba U-Net model
  measure_underwater.py  - PSNR, SSIM, CIEDE2000, UCIQE, UIQM metrics
  train.py               - Main training loop
  eval.py                - Checkpoint evaluation
  net_test.py            - Model profiling
  PHYSICS_PRIORS.md      - Physics prior documentation
  requirements.txt
  README.md
```

---

## Runtime Notes

Physics maps are currently computed on the fly in the DataLoader collate path.
`gdcp` is significantly slower than `udcp` and `gupdm` because it performs
full-image depth estimation, morphology, least-squares fitting, and filtering.

For quick debugging, use:

```bash
--prior_method udcp
```

For Ucolor-style experiments, use:

```bash
--prior_method gdcp
```

For long training runs with heavy priors, precomputing or caching physics maps
is recommended.

---

## Results

| Variant | Prior | Val PSNR | Test PSNR | Val SSIM | Test SSIM |
|---|---|---:|---:|---:|---:|
| `unet_3ch` | none | - | - | - | - |
| `unet_5ch` | `udcp` | - | - | - | - |
| `unet_5ch` | `gdcp` | - | - | - | - |
| `unet_5ch` | `gupdm` | - | - | - | - |

Fill in after training.
