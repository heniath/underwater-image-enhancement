# UW-LYT design and lightweight literature review

UW-LYT adapts LYT-Net's fixed luminance/chrominance split to underwater
enhancement while using LU2Net-style axial depthwise convolution and a small
low-resolution attention path. It predicts an identity-initialized RGB
residual. The RGB model is the deployment target; legacy 4/5-channel variants
project the runner-generated prior maps through a separate 1x1 branch.

## Efficiency evidence

Published efficiency numbers are not directly comparable unless input size,
operation-count convention, software, and hardware match. A dash means the
primary source did not provide a sufficiently clear value. Parameters, FLOPs,
preprocessing, memory, and measured latency are therefore kept in separate
columns instead of treating “lightweight” as a single number.

| Model | Task | Parameters | FLOPs and input | External preprocessing | Reported memory | Measured hardware latency |
|---|---|---:|---:|---|---:|---|
| [LYT-Net](https://arxiv.org/abs/2401.15204) | Low light | 45k | 3.49 G (paper configuration) | Fixed RGB→YUV | — | — |
| [FA⁺Net](https://arxiv.org/abs/2305.08824) | Underwater | 8.974k | — | Strong-prior stage is part of the network | 0.03 MB model size in a later unified study | Paper reports about 0.01 s; a later 640×480 study measures 11.7 ms GPU / 423.43 ms mobile SoC |
| [LSNet](https://arxiv.org/abs/2405.16197) | Underwater | 7.534k | — | Transmission-map-inspired mechanism; no separately timed external prior reported | 0.03 MB model size in a later unified study | Later 640×480 study: 59.519 ms GPU / >500 ms mobile SoC |
| [Zero-UAE](https://doi.org/10.3389/fmars.2024.1378817) | Underwater, zero reference | 17,699 | 1.15 G at 256² | Learned curve estimation; no external image prior | 128.75 MB reported resource occupancy | — |
| [LU2Net](https://arxiv.org/abs/2406.14973) | Underwater | — | — | None reported | — | RTX 3060 laptop: 100 fps; i7-10750H CPU: 12 fps |
| [LPS-Net](https://doi.org/10.3390/app13169419) | Underwater | 80.12k | 99.02 G at 1280×720 | None reported | — | RTX 3090 timing is reported in its efficiency table; retain the paper value only with that resolution/hardware |
| [LiteEnhanceNet](https://doi.org/10.1016/j.eswa.2023.122546) | Underwater | 13.688k | — | None reported | 0.05 MB model size in a later unified study | Later 640×480 study: 6.654 ms GPU / 190.02 ms mobile SoC |
| [MobileIE](https://openaccess.thecvf.com/content/ICCV2025/html/Yan_MobileIE_An_Extremely_Lightweight_and_Effective_ConvNet_for_Real-Time_Image_ICCV_2025_paper.html) | General/mobile enhancement | 4.047k | 0.924 G at 640×480 | None | 0.02 MB model size | 0.910 ms GPU / 8.94 ms mobile SoC at 640×480 |

The later unified 640×480 measurements cited above come from MobileIE's ICCV
2025 comparison table. They are useful deployment context, not substitutes for
matched measurements in this repository.

## Matched protocol and acceptance rule

The primary experiment uses seeds 0, 1, and 2; 50 epochs; batch 16; 256²
crops; Adam at 1e-4 with weight decay 1e-5; StepLR(30, 0.5); L1/perceptual/SSIM
weights 1/1/0; patience 20; and best validation PSNR checkpoint selection.
EUVP uses UDCP. UIEB retains the runner's existing prior list.

The JSON report accepts `uwlyt_3ch` when mean PSNR is at least the matched
legacy mean minus 0.5 dB and mean SSIM is at least the matched mean minus 0.01.
Reference aggregates are EUVP 20.896 dB / 0.8857 and UIEB 21.017 dB / 0.8871.
Balanced losses, cosine schedules, distillation, and longer training belong in
separately labelled training-recipe ablations.

Profile all UW-LYT variants with:

```bash
uwir-profile uwlyt uwlyttiny --device cpu --no-pretrained --img-size 256 --runs 20
uwir-profile uwlyt uwlyttiny --device cuda --no-pretrained --img-size 256 --runs 20
```

The profiler records parameter count, FLOPs when `thop` is installed, peak
memory, batch-one CPU/CUDA latency, and a latency-scope field. For physics
variants, the reported end-to-end latency includes UDCP prior generation.
