# Physics-Informed Underwater Image Restoration

This repository contains the implementation used to study whether transmission
and backscatter priors improve underwater image restoration when the restoration
backbone is held fixed. The published study uses one customized U-Net with four
input configurations:

| Model | Input |
|---|---|
| `unet_3ch` | RGB |
| `unet_4ch_t` | RGB + transmission map |
| `unet_4ch_b` | RGB + background/backscatter map |
| `unet_5ch` | RGB + both physics maps |

UW-LYT and UW-LYT-Tiny are deliberately retained as ongoing lightweight-model
experiments. Their results are not mixed with the paper's reported U-Net
results.

## Repository scope

```text
src/uwir/
  cli/          training, evaluation, and profiling commands
  data/         paired UIEB and EUVP loaders
  models/       customized U-Net and UW-LYT
  physics/      paper physics-channel implementation
  losses.py     L1, perceptual, and optional SSIM losses
  metrics.py    PSNR, SSIM, CIEDE2000, UCIQE, and UIQM
scripts/
  experiments/  multi-seed UIEB and EUVP ablations
  visualization/ paper-result plotting
tests/          focused reproducibility and model tests
```

The manuscript source is intentionally not stored in this repository. Reported
numeric results are preserved separately in
[`docs/reported_results.md`](docs/reported_results.md).

## Installation

Python 3.10 or newer is required.

```bash
python -m pip install -e .
python -m pip install -e '.[dev,profile,visualization]'
```

## Datasets

```text
datasets/
  EUVP/
    Paired/
      underwater_imagenet/trainA/ trainB/
      underwater_dark/trainA/ trainB/
      underwater_scenes/trainA/ trainB/
    test_samples/Inp/ GTr/
  UIEB/
    raw-890/
    reference-890/
```

UIEB uses 800 development pairs and a held-out T90 evaluation set. EUVP training
combines its three paired subsets, and evaluation uses the 515-image official
test sample collection.

## Paper protocol

The paper reports three independent runs per input configuration, 100 epochs,
batch size 4, random 256×256 crops, Adam with learning rate `2e-4`, and cosine
annealing. It evaluates PSNR, SSIM, CIEDE2000, UCIQE, and UIQM.

Train one configuration:

```bash
uwir-train \
  --model unet_5ch \
  --dataset euvp \
  --data_train_euvp ./datasets/EUVP \
  --nEpochs 100 \
  --batchSize 4 \
  --cropSize 256 \
  --lr 2e-4 \
  --cos_restart true \
  --L1_weight 1.0 \
  --perceptual_weight 1.0 \
  --SSIM_weight 0.0 \
  --seed 0
```

Run the multi-seed experiment drivers:

```bash
python -m scripts.experiments.ablation_euvp --variants \
  unet_3ch unet_4ch_t unet_4ch_b unet_5ch

python -m scripts.experiments.ablation_uieb --variants \
  unet_3ch unet_4ch_t unet_4ch_b unet_5ch
```

All runner settings can be overridden explicitly. Existing reported values are
historical paper results and are never overwritten by the documentation.

Evaluate compatible checkpoint directories:

```bash
uwir-evaluate \
  --eval_benchmark euvp \
  --data_train_euvp ./datasets/EUVP \
  --checkpoint_dir ./checkpoints \
  --val_folder ./results/euvp
```

## UW-LYT experiments

UW-LYT supports the same input contracts as the U-Net, allowing a future
controlled lightweight comparison:

```bash
uwir-train --model uwlyt_3ch --dataset euvp
uwir-train --model uwlyt_4ch_t --dataset euvp
uwir-train --model uwlyt_4ch_b --dataset euvp
uwir-train --model uwlyt_5ch --dataset euvp
```

Use `uwlyttiny_*` for the narrower Tiny variant. Profile retained architectures
with:

```bash
uwir-profile --list
uwir-profile uwlyt --device cuda
```

## Tests

```bash
pytest
```

## License

See [LICENSE](LICENSE).
