# Reference-method UIEB/LSUI benchmark

This benchmark is a set of **method-preserving reference reruns under a standardized data/split/evaluation protocol**. It is not the physics ablation, a modern-architecture screen, a unified-loss architecture benchmark, or an exact reproduction of the original publications.

## Fixed outer protocol

- Methods: FUnIE-GAN, UColor, the paper/project U-Net, Water-Net, UWFormer.
- Datasets: UIEB and LSUI.
- Model seeds: `0, 1, 2`; split seed: `42`.
- 100 epochs, paired 256×256 crops, effective batch size 4.
- Validate each epoch and select `best_model.pth` by validation PSNR.
- Continue through epoch 100, reload the selected checkpoint, then evaluate the held-out test set once.
- Quality metrics: PSNR, SSIM, CIEDE2000, UCIQE, and UIQM through one guarded evaluator.
- Dataset images are decoded into RAM when the estimated decoded size is at most half of currently available memory. Pass `--no-ram-cache` to disable this.

Method architectures, required derived inputs, losses, optimizers, schedulers, auxiliary networks, and output conventions remain adapter-specific. See `outputs/reference_methods/method_provenance.json` and `method_training_configs.json` after preflight.

## Data roots

The default is resolved from the checkout, not a machine-specific absolute path:

```text
<repository>/datasets/UIEB/raw-890
<repository>/datasets/UIEB/reference-890
<repository>/datasets/LSUI/...
```

On the SSH checkout, either run from `/mnt/disk2/student/DE200277_TruongDKN/underwater-image-enhancement` or point the smaller checkout at a data root explicitly with `--data-root`. LSUI discovery recursively prints a compact tree and refuses ambiguous pairing.

## Launch

Install the checkout first:

```bash
python -m pip install -e '.[dev,profile,visualization]'
```

Run the required 10-combination smoke matrix (seed 0, one epoch, minimal batches/samples):

```bash
python -m scripts.reference_methods_benchmark --smoke
```

Run/resume all 30 experiments only after the complete smoke matrix passes:

```bash
python -m scripts.reference_methods_benchmark --full
```

Select a mounted SSH data directory without embedding it in source:

```bash
python -m scripts.reference_methods_benchmark \
  --smoke \
  --data-root /mnt/disk2/student/DE200277_TruongDKN/underwater-image-enhancement/datasets
```

The notebook `notebooks/reference_methods_uieb_lsui.ipynb` exposes the same flow interactively. `RUN_FULL` and `RUN_CROSS_DATASET` default to false.

## Resume and output safety

Every epoch atomically updates `last_model.pth` with all model, optimizer, scheduler, scaler, history, method state, and run metadata. `best_model.pth` stores the same complete state at the highest validation PSNR. An incomplete run resumes at the next epoch; a run with completed test metrics is skipped. Smoke checkpoints live below `outputs/reference_methods/smoke/`, so they cannot be mistaken for full-run checkpoints.

No existing notebooks, physics results, or historical architecture results are overwritten.

## Known reproduction boundaries

- FUnIE-GAN is an attributed PyTorch/NCHW port of the authors' Keras paired `generator2` and conditional discriminator, retaining adversarial and content training.
- UColor's official GitHub repository points to a separate multi-gigabyte TensorFlow bundle. The local PyTorch implementation follows the published RGB/HSV/Lab, channel-attention, and GDCP-guided formulation and records the supporting reimplementation commit. It is not checkpoint-compatible with the official TensorFlow release.
- Water-Net is an attributed PyTorch port of the authors' TensorFlow 1.x topology. OpenCV CLAHE replaces MATLAB `adapthisteq`, which is close but not bit-identical.
- The U-Net is the exact implementation retained by this project at the recorded repository commit.
- UWFormer preserves the released Haar low/high-frequency split, transformer/Fourier branches, and released supervised objective. The dependency-free low-frequency port is not parameter-for-parameter checkpoint compatible with upstream LPViT/NAFNet; the released code/paper semi-supervision discrepancy is recorded.

These boundaries are why the results must not be described as exact publication reproductions.
