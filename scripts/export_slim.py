"""
scripts/export_slim.py
======================
Convert a trained multi-branch FGDPA checkpoint into a re-parameterized slim model
for ultra-fast inference (4.234K parameters, 800+ FPS).
"""

import argparse
from pathlib import Path
import torch

from uwir.models import build_model
from uwir.cli.train import load_ckpt

def main():
    parser = argparse.ArgumentParser(description="Export FGDPA to re-parameterized slim checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best_model.pth")
    parser.add_argument("--model", type=str, default="fgdpa_3ch", choices=["fgdpa_3ch", "fgdpa_5ch"])
    parser.add_argument("--output", type=str, default="", help="Output path for slim checkpoint")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    out_path = Path(args.output) if args.output else ckpt_path.parent / "best_model_slim.pth"

    print(f"Loading training model: {args.model} from {ckpt_path}")
    model = build_model(args.model, pretrained_backbone=False)
    load_ckpt(str(ckpt_path), model)
    model.eval()

    print("Re-parameterizing multi-branch convolutions into single Conv2d layers...")
    model_slim = model.slim().eval()

    params_orig = sum(p.numel() for p in model.parameters())
    params_slim = sum(p.numel() for p in model_slim.parameters())

    torch.save({"model": model_slim.state_dict()}, str(out_path))
    # Also save raw state_dict for compatibility
    raw_path = ckpt_path.parent / "model_best_slim.pkl"
    torch.save(model_slim.state_dict(), str(raw_path))

    print(f"[SUCCESS] Exported slim model to:")
    print(f"  - Package format: {out_path}")
    print(f"  - Raw state_dict: {raw_path}")
    print(f"Parameters: {params_orig:,} -> {params_slim:,} ({params_slim / 1000:.3f}K)")

if __name__ == "__main__":
    main()
