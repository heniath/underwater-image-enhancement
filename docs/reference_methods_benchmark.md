# Reference-method UIEB/LSUI benchmark

This benchmark is a set of **method-preserving reference reruns under a standardized data/split/evaluation protocol**. It is not the physics ablation, a modern-architecture screen, a unified-loss architecture benchmark, or an exact reproduction of the original publications.

## Fixed outer protocol

- Reference methods: FUnIE-GAN, UColor, the paper/project U-Net, Water-Net, UWFormer.
- Supplemental requested method: LPD-Net, recorded separately as a paper-guided faithful reimplementation.
- Execution priority: methods with released or repository-backed source run first; paper-only reimplementations run afterward. The default order is FUnIE-GAN, UColor, U-Net, Water-Net, UWFormer, then LPD-Net.
- Datasets: UIEB and LSUI.
- Model seeds: `0, 1, 2`; split seed: `42`.
- 100 epochs, paired 256×256 crops, effective batch size 4.
- Validate each epoch and select `best_model.pth` by validation PSNR.
- Continue through epoch 100, reload the selected checkpoint, then evaluate the held-out test set once.
- Quality metrics: PSNR, SSIM, CIEDE2000, UCIQE, and UIQM through one guarded evaluator.
- Dataset images are decoded into RAM when the estimated decoded size is at most half of currently available memory. Pass `--no-ram-cache` to disable this.

Method architectures, required derived inputs, losses, optimizers, schedulers, auxiliary networks, and output conventions remain adapter-specific. See `outputs/reference_methods/method_provenance.json` and `method_training_configs.json` after preflight.

## Source-code priority

| Tier | Method | Released source used as authority | Local integration boundary |
|---|---|---|---|
| 1 | FUnIE-GAN | [Author PyTorch repository](https://github.com/xahidbuffon/FUnIE-GAN) | Native released topology, initialization, loss, and optimizer wrapped by the common adapter |
| 1 | UColor | [Author repository](https://github.com/Li-Chongyi/Ucolor) and [PyTorch reimplementation](https://github.com/CV-Reimplementation/Ucolor-Reimplementation) | Official release is a separately hosted TensorFlow bundle; the local PyTorch adapter is not checkpoint-compatible |
| 1 | U-Net | This repository | Project baseline reused directly |
| 1 | Water-Net | [Author repository](https://github.com/Li-Chongyi/Water-Net_Code) | Released TensorFlow 1.x network is ported to the common PyTorch adapter |
| 1 | UWFormer | [Author repository](https://github.com/leiyingtie/UWFormer) | Released formulation is adapted without its external wavelet/attention dependencies; it is not checkpoint-compatible |
| 2 | LPD-Net | No source repository located | Paper-guided local reconstruction; supplemental only |

Tier 1 is the default execution queue and Tier 2 follows it. A GitHub link alone does not imply checkpoint compatibility; the final column records where framework or dependency adaptation remains necessary.

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

Run the required 12-combination smoke matrix (seed 0, one epoch, minimal batches/samples):

```bash
python -m scripts.reference_methods_benchmark --smoke
```

Run/resume the 30 reference runs plus 6 supplemental LPD-Net runs only after the complete smoke matrix passes:

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

- FUnIE-GAN uses the authors' released native PyTorch five-level U-Net generator, conditional discriminator, initialization, losses, and optimizer settings behind the common benchmark adapter.
- UColor's official GitHub repository points to a separate multi-gigabyte TensorFlow bundle. The local PyTorch implementation follows the published RGB/HSV/Lab, channel-attention, and GDCP-guided formulation and records the supporting reimplementation commit. It is not checkpoint-compatible with the official TensorFlow release.
- Water-Net is an attributed PyTorch port of the authors' TensorFlow 1.x topology. OpenCV CLAHE replaces MATLAB `adapthisteq`, which is close but not bit-identical.
- The U-Net is the exact implementation retained by this project at the recorded repository commit.
- UWFormer preserves the released Haar low/high-frequency split, transformer/Fourier branches, and released supervised objective. The dependency-free low-frequency port is not parameter-for-parameter checkpoint compatible with upstream LPViT/NAFNet; the released code/paper semi-supervision discrepancy is recorded.
- LPD-Net has no source repository identified by the supplied paper or current source search. The local 1.843M-parameter implementation preserves the documented MSRCR/ANGF/HDA/LKA/CSC components and four-term loss, but missing channel, block, MSRCR, weight-decay, and differentiable-UCIQE details are local documented choices. It is supplemental and must not be presented as an official reproduction.

These boundaries are why the results must not be described as exact publication reproductions.
