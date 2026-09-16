"""
Underwater Image Restoration — Training Options
Replaces the CIDNet low-light options with UWIR-specific arguments.
"""

import argparse

from uwir.models import ALL_MODEL_NAMES


def _str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def option():
    parser = argparse.ArgumentParser(description="UWIR — Physics-Guided Underwater Image Restoration")

    # ------------------------------------------------------------------
    # Core training hyper-parameters
    # ------------------------------------------------------------------
    parser.add_argument(
        "--batchSize",
        type=int,
        default=16,
        help="Training mini-batch size",
    )
    parser.add_argument(
        "--cropSize",
        type=int,
        default=256,
        help="Resize target size for training images (height=width)",
    )
    parser.add_argument(
        "--nEpochs",
        type=int,
        default=200,
        help="Total number of training epochs",
    )
    parser.add_argument(
        "--start_epoch",
        type=int,
        default=0,
        help="Starting epoch (> 0 resumes from checkpoint)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="Path to a checkpoint .pth to resume training from "
        "(overrides --start_epoch logic). Example: "
        "--resume ./checkpoints/run/epoch_0040.pth",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="",
        help="Human-readable prefix for the checkpoint subdirectory. "
        "A timestamp is always appended: <run_name>_<YYYYMMDD_HHMMSS>. "
        "Omit to use <model>_<dataset>_<YYYYMMDD_HHMMSS>. "
        "Example: --run_name unet5ch_euvp_run1",
    )
    parser.add_argument(
        "--snapshots", type=int, default=10, help="Save a checkpoint every N epochs"
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate (Adam)")
    parser.add_argument(
        "--weight_decay", type=float, default=1e-5, help="Adam weight decay"
    )
    parser.add_argument("--gpu_mode", type=_str2bool, default=True)
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="Number of GPUs to use via DataParallel (1 = single GPU)",
    )
    parser.add_argument("--shuffle", type=_str2bool, default=True)
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="DataLoader worker threads (keep low to avoid OOM on Kaggle; 2-4 recommended)",
    )
    parser.add_argument(
        "--in_memory",
        type=_str2bool,
        default=False,
        help="Load all images into RAM during initialization",
    )

    # ------------------------------------------------------------------
    # Learning-rate scheduler
    # ------------------------------------------------------------------
    parser.add_argument(
        "--cos_restart",
        type=_str2bool,
        default=False,
        help="Use CosineAnnealingWarmRestarts scheduler",
    )
    parser.add_argument(
        "--cos_restart_cyclic",
        type=_str2bool,
        default=False,
        help="Use cyclic cosine restart variant",
    )
    parser.add_argument(
        "--scheduler_step",
        type=int,
        default=30,
        help="StepLR step size in epochs",
    )
    parser.add_argument(
        "--scheduler_gamma",
        type=float,
        default=0.5,
        help="StepLR decay factor",
    )
    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=0,
        help="Number of linear warm-up epochs",
    )
    parser.add_argument(
        "--start_warmup",
        type=_str2bool,
        default=False,
        help="Enable warm-up at the start of training",
    )

    # ------------------------------------------------------------------
    # Early stopping
    # ------------------------------------------------------------------
    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=20,
        help="Stop training if val SSIM does not improve for this many epochs (proposal §4.5)",
    )
    parser.add_argument(
        "--val_interval",
        type=int,
        default=1,
        help="Compute validation metrics every N epochs; early stopping advances only then",
    )

    # ------------------------------------------------------------------
    # Physics front-end
    # ------------------------------------------------------------------
    parser.add_argument(
        "--prior_method",
        type=str,
        default="udcp",
        choices=["udcp"],
        help="Paper transmission/background prior implementation",
    )
    parser.add_argument(
        "--guided_filter_radius",
        type=int,
        default=15,
        help="Radius for the guided image filter used to refine the transmission map",
    )
    parser.add_argument(
        "--guided_filter_eps",
        type=float,
        default=1e-3,
        help="Regularisation epsilon for the guided filter",
    )

    # ------------------------------------------------------------------
    # Model / ablation variant
    # ------------------------------------------------------------------
    parser.add_argument(
        "--model",
        type=str,
        default="unet_5ch",
        choices=ALL_MODEL_NAMES,
        help=(
            "Model variant (backbone_channels):\n"
            "  Channels: 3ch=RGB only | 4ch_t=RGB+t(x) | 4ch_b=RGB+B | 5ch=RGB+t(x)+B\n"
            "  V2: 6ch_b=RGB+B_RGB | 7ch=RGB+t(x)+B_RGB\n"
            "  Backbones: unet | uwlyt | uwlyttiny | uwlytv2 | uwlytv2tiny"
        ),
    )
    parser.add_argument(
        "--pretrained_backbone",
        type=_str2bool,
        default=True,
        help="Compatibility flag; retained models do not use pretrained backbones",
    )

    # ------------------------------------------------------------------
    # Loss weights  λ1·L_pixel + λ2·L_perceptual + λ3·L_SSIM
    # ------------------------------------------------------------------
    parser.add_argument(
        "--L1_weight",
        type=float,
        default=1.0,
        help="λ1 — pixel-wise MAE loss weight",
    )
    parser.add_argument(
        "--perceptual_weight",
        type=float,
        default=1.0,
        help="λ2 — VGG-16 perceptual loss weight",
    )
    parser.add_argument(
        "--SSIM_weight",
        type=float,
        default=0.0,
        help="λ3 — SSIM loss weight",
    )

    # ------------------------------------------------------------------
    # Training dataset paths
    # ------------------------------------------------------------------
    parser.add_argument(
        "--data_train_euvp",
        type=str,
        default="./datasets/EUVP",
        help="Root of EUVP release (primary training corpus)",
    )
    parser.add_argument(
        "--euvp_subset",
        type=str,
        default="all",
        help=(
            "EUVP sub-set(s) to use for training.\n"
            '  "all"              — all three subsets (notebook default)\n'
            '  "underwater_imagenet" | "underwater_dark" | "underwater_scenes"\n'
            '  Comma-separated for multiple: "underwater_dark,underwater_scenes"'
        ),
    )
    parser.add_argument(
        "--data_train_uieb",
        type=str,
        default="./datasets/UIEB",
        help="Root of UIEB release (supplementary training)",
    )

    # ------------------------------------------------------------------
    # Validation / evaluation input paths
    # ------------------------------------------------------------------
    parser.add_argument(
        "--data_val_uieb",
        type=str,
        default="./datasets/UIEB/test/input",
        help="UIEB test-90 input images",
    )
    parser.add_argument(
        "--data_val_euvp",
        type=str,
        default="./datasets/EUVP/Paired/underwater_imagenet/validation",
        help="EUVP unpaired validation images (no GT available)",
    )

    # ------------------------------------------------------------------
    # Validation / evaluation ground-truth paths
    # ------------------------------------------------------------------
    parser.add_argument(
        "--data_valgt_uieb",
        type=str,
        default="./datasets/UIEB/test/reference",
        help="UIEB test-90 reference (ground-truth) images",
    )
    parser.add_argument(
        "--data_valgt_euvp",
        type=str,
        default="",
        help="EUVP ground-truth (empty: EUVP validation is unpaired, no GT)",
    )

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    parser.add_argument(
        "--val_folder",
        type=str,
        default="./results/",
        help="Directory for saving validation output images",
    )
    parser.add_argument(
        "--native_eval",
        type=_str2bool,
        default=True,
        help="Evaluate at native resolution in addition to the legacy square resize",
    )
    parser.add_argument(
        "--eval_benchmark",
        choices=["euvp", "uieb"],
        default="euvp",
        help="Paired benchmark evaluated by uwir-evaluate",
    )
    parser.add_argument("--tile_size", type=int, default=512)
    parser.add_argument("--tile_overlap", type=int, default=64)
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./checkpoints/",
        help="Directory for saving model checkpoints",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="./logs/",
        help="Directory for training logs",
    )

    # ------------------------------------------------------------------
    # Misc / reproducibility
    # ------------------------------------------------------------------
    parser.add_argument(
        "--seed", type=int, default=42, help="Global random seed for reproducibility"
    )
    parser.add_argument(
        "--split_seed",
        type=int,
        default=42,
        help="Fixed data-split seed, independent of the model-training seed",
    )
    parser.add_argument(
        "--grad_clip",
        type=_str2bool,
        default=True,
        help="Enable gradient clipping to stabilise training",
    )
    parser.add_argument(
        "--grad_detect",
        type=_str2bool,
        default=False,
        help="Enable anomaly detection (slow; use for debugging only)",
    )
    parser.add_argument(
        "--amp",
        type=_str2bool,
        default=True,
        help="Use automatic mixed precision on CUDA",
    )
    parser.add_argument(
        "--grad_accumulation_steps",
        type=int,
        default=1,
        help="Accumulate this many mini-batches before each optimizer step",
    )

    # ------------------------------------------------------------------
    # Dataset selector (controls which loader is used in train.py)
    # ------------------------------------------------------------------
    parser.add_argument(
        "--dataset",
        type=str,
        default="euvp",
        choices=["euvp", "uieb", "euvp+uieb"],
        help=(
            "Training dataset:\n"
            "  euvp       — EUVP only  (primary, proposal §4.5)\n"
            "  uieb       — UIEB only\n"
            "  euvp+uieb  — Combined EUVP + UIEB"
        ),
    )

    return parser
