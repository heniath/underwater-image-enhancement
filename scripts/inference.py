"""
scripts/inference.py
====================
Run inference on underwater images using a trained checkpoint (SGMA-Net, UNet, UW-LYT, etc.)
and save the enhanced output images to disk.
"""

import argparse
import os
import sys
from pathlib import Path
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from tqdm import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from uwir.cli.evaluate import _parse_model_name
from uwir.cli.train import _add_physics_channels, _resolve_physics_extractor, load_ckpt
from uwir.metrics import tiled_predict
from uwir.models import ALL_MODEL_NAMES, build_model, parse_model_variant

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPG", ".JPEG", ".PNG"}


def parse_args():
    parser = argparse.ArgumentParser(description="UWIR Image Enhancement Inference")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint file (e.g. best_model.pth) or checkpoint directory",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="",
        choices=[""] + ALL_MODEL_NAMES,
        help="Model variant name (e.g. sgmanet_5ch). If omitted, inferred from checkpoint name",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="",
        help="Directory of images to enhance",
    )
    parser.add_argument(
        "--input_image",
        type=str,
        default="",
        help="Single image path to enhance",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results/inference_outputs",
        help="Directory where enhanced images will be saved",
    )
    parser.add_argument(
        "--prior_method",
        type=str,
        default="udcp",
        help="Physics prior method (default: udcp)",
    )
    parser.add_argument(
        "--resize",
        type=int,
        default=0,
        help="Optional square resize (0 = keep original resolution)",
    )
    parser.add_argument(
        "--tile_size",
        type=int,
        default=512,
        help="Tile size for native high-res tiled inference (0 to disable tiling)",
    )
    parser.add_argument(
        "--tile_overlap",
        type=int,
        default=64,
        help="Tile overlap in pixels",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on (cuda or cpu)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Resolve checkpoint path
    ckpt_path = Path(args.checkpoint)
    if ckpt_path.is_dir():
        candidate = ckpt_path / "best_model.pth"
        if candidate.is_file():
            ckpt_file = candidate
            run_name = ckpt_path.name
        else:
            raise FileNotFoundError(f"No best_model.pth found in {ckpt_path}")
    elif ckpt_path.is_file():
        ckpt_file = ckpt_path
        run_name = ckpt_path.parent.name
    else:
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # Resolve model variant
    model_name = args.model or _parse_model_name(run_name)
    if not model_name:
        raise ValueError(
            f"Could not infer model variant from '{run_name}'. Please specify --model explicitly."
        )

    print(f"Model variant: {model_name}")
    print(f"Checkpoint   : {ckpt_file}")

    _, in_channels, physics_mode = parse_model_variant(model_name)
    physics_extractor = _resolve_physics_extractor(args.prior_method)

    # Build model & load weights
    model = build_model(model_name, pretrained_backbone=False).to(device)
    ckpt_epoch, ckpt_metrics = load_ckpt(str(ckpt_file), model, device=str(device))
    print(f"Loaded epoch : {ckpt_epoch} (stored metrics: {ckpt_metrics})")
    model.eval()

    # Collect input images
    input_files = []
    if args.input_image:
        input_files.append(Path(args.input_image))
    if args.input_dir:
        dir_path = Path(args.input_dir)
        if not dir_path.is_dir():
            raise NotADirectoryError(f"Directory not found: {dir_path}")
        for p in sorted(dir_path.iterdir()):
            if p.suffix in IMG_EXTS:
                input_files.append(p)

    if not input_files:
        raise ValueError("No input images found! Provide --input_dir or --input_image.")

    print(f"Found {len(input_files)} image(s) to process.")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    to_tensor = transforms.ToTensor()

    with torch.no_grad():
        for img_path in tqdm(input_files, desc="Enhancing"):
            img = Image.open(img_path).convert("RGB")
            if args.resize > 0:
                img = img.resize((args.resize, args.resize), Image.Resampling.BILINEAR)

            rgb_tensor = to_tensor(img)  # (3, H, W) in [0, 1]

            # Add physics channels if required
            inp_tensor = _add_physics_channels(rgb_tensor, physics_mode, physics_extractor)
            inp_tensor = inp_tensor.unsqueeze(0).to(device)  # (1, C, H, W)

            # Inference
            if args.tile_size and (
                inp_tensor.shape[2] > args.tile_size or inp_tensor.shape[3] > args.tile_size
            ):
                out = tiled_predict(
                    model,
                    inp_tensor,
                    tile_size=args.tile_size,
                    overlap=args.tile_overlap,
                )
            else:
                out = model(inp_tensor)

            out = out.squeeze(0).clamp(0.0, 1.0).cpu()
            save_path = output_dir / img_path.name
            save_image(out, str(save_path))

    print(f"\nAll enhanced images saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
