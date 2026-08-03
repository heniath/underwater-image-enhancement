# Reported paper results

These values reproduce the numbers reported in the paper. They are retained as
an immutable reference and are not recalculated during tests.

## Physics-informed input ablation

### EUVP (515 images)

| Model | Ch. | PSNR ↑ | SSIM ↑ | CIEDE2000 ↓ | UCIQE ↑ | UIQM ↑ |
|---|---:|---:|---:|---:|---:|---:|
| UNet-3ch | 3 | 26.458 ± 0.238 | **0.8729 ± 0.002** | 5.489 ± 0.162 | 26.623 ± 0.079 | 1.797 ± 0.009 |
| UNet-4ch-t | 4 | 26.422 ± 0.179 | 0.8713 ± 0.001 | 5.505 ± 0.088 | 26.629 ± 0.318 | **1.801 ± 0.023** |
| UNet-4ch-B | 4 | 26.532 ± 0.260 | 0.8726 ± 0.002 | 5.394 ± 0.126 | **26.631 ± 0.251** | 1.798 ± 0.016 |
| UNet-5ch | 5 | **26.644 ± 0.034** | 0.8726 ± 0.001 | **5.356 ± 0.018** | 26.459 ± 0.094 | 1.787 ± 0.006 |

### UIEB T90 (90 images)

| Model | Ch. | PSNR ↑ | SSIM ↑ | CIEDE2000 ↓ | UCIQE ↑ | UIQM ↑ |
|---|---:|---:|---:|---:|---:|---:|
| UNet-3ch | 3 | 20.896 ± 0.068 | 0.8857 ± 0.0015 | 10.785 ± 0.054 | 27.036 ± 0.211 | 1.631 ± 0.010 |
| UNet-4ch-t | 4 | 21.067 ± 0.180 | **0.8875 ± 0.0014** | 10.642 ± 0.102 | **27.246 ± 0.071** | **1.642 ± 0.003** |
| UNet-4ch-B | 4 | **21.207 ± 0.096** | 0.8861 ± 0.0008 | **10.513 ± 0.126** | 27.039 ± 0.304 | 1.624 ± 0.024 |
| UNet-5ch | 5 | 21.012 ± 0.198 | 0.8835 ± 0.0038 | 10.730 ± 0.154 | 27.008 ± 0.233 | 1.609 ± 0.029 |

## Representative-method comparison

| Method | T90 PSNR | T90 SSIM | EUVP-Test PSNR | EUVP-Test SSIM |
|---|---:|---:|---:|---:|
| FUnIE-GAN | 19.18 | 0.865 | 26.190 | 0.740 |
| UColor | 20.81 | 0.904 | 20.670 | 0.786 |
| U-Net | — | — | 22.190 | 0.802 |
| Water-Net | 20.87 | **0.911** | 24.430 | 0.820 |
| UWFormer | — | — | 24.400 | 0.845 |
| Ours (5-ch) | **21.012** | 0.884 | **26.644** | **0.873** |

UW-LYT remains an ongoing experiment and therefore has no value in these paper
tables.
