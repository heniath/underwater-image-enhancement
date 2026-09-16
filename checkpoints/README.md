# Model Checkpoints

This directory is used to store trained PyTorch model checkpoints (`.pth`).
Model weights are large binary files (>100MB) and are excluded from git version control.

## Where to Place Weights

Place trained checkpoint files or subfolders here:

```text
checkpoints/
├── best_model.pth                         # Single model weights file
└── sgmanet_5ch_euvp/                      # Or experiment run folder
    ├── best_model.pth
    └── config.json
```

## Running Inference with a Checkpoint

```bash
python scripts/inference.py \
    --checkpoint checkpoints/best_model.pth \
    --model sgmanet_5ch \
    --input_dir path/to/raw_images \
    --output_dir results/enhanced_images
```

## Hosting Checkpoints

Recommended places to host pretrained weights for sharing:
- **GitHub Releases** (supports files up to 2GB attached to release tags)
- **Hugging Face Hub**
- **Google Drive / OneDrive**
