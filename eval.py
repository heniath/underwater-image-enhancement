import argparse
import importlib.util
import json
import os
import time

import numpy as np
import torch
import torch.utils.data as data

from data.util import is_image_file, load_img
from measure_underwater import evaluate_loader


ALL_MODEL_NAMES = [
    f"{backbone}_{variant}"
    for backbone in ("unet", "resnet", "mobilenet", "mambavision", "mambaunet")
    for variant in ("3ch", "4ch_t", "4ch_b", "5ch")
]

_VARIANTS = {
    "3ch": (3, "none"),
    "4ch_t": (4, "t"),
    "4ch_b": (4, "b"),
    "5ch": (5, "tb"),
}


def parse_model_variant(name):
    for backbone in ("unet", "resnet", "mobilenet", "mambavision", "mambaunet"):
        prefix = backbone + "_"
        if name.startswith(prefix):
            variant = name[len(prefix):]
            if variant not in _VARIANTS:
                raise ValueError(f"Unknown model variant: {name}")
            in_channels, physics_mode = _VARIANTS[variant]
            return backbone, in_channels, physics_mode
    raise ValueError(f"Unknown model name: {name}")


def _load_module_from_file(module_name, relative_path):
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, relative_path)
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_model_lazy(name, pretrained_backbone=False):
    backbone, in_channels, _ = parse_model_variant(name)

    if backbone == "unet":
        unet_module = _load_module_from_file("eval_unet_module", os.path.join("net", "unet.py"))
        return unet_module.UNet5ch(in_channels=in_channels)

    from net.registry import build_model

    return build_model(name, pretrained_backbone=pretrained_backbone)


def compute_physics_maps_lazy(image_np):
    physics_module = _load_module_from_file(
        "eval_physics_module",
        os.path.join("net", "physics.py"),
    )
    return physics_module.compute_physics_maps(image_np)


def _add_physics_channels(rgb_tensor, mode):
    if mode == "none":
        return rgb_tensor

    img_np = rgb_tensor.permute(1, 2, 0).numpy().astype(np.float32)
    t_map, b_map = compute_physics_maps_lazy(img_np)

    if mode == "t":
        return torch.cat([rgb_tensor, torch.from_numpy(t_map).unsqueeze(0)], dim=0)
    if mode == "b":
        return torch.cat([rgb_tensor, torch.from_numpy(b_map).unsqueeze(0)], dim=0)
    if mode == "tb":
        return torch.cat([
            rgb_tensor,
            torch.from_numpy(t_map).unsqueeze(0),
            torch.from_numpy(b_map).unsqueeze(0),
        ], dim=0)
    raise ValueError(f"Unknown physics mode: {mode}")


def _unwrap(model):
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def load_ckpt(path, model, optimizer=None, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    _unwrap(model).load_state_dict(state_dict)
    if optimizer and isinstance(ckpt, dict) and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    epoch = ckpt.get("epoch", 0) if isinstance(ckpt, dict) else 0
    metrics = ckpt.get("metrics", {}) if isinstance(ckpt, dict) else {}
    return epoch, metrics


class PairedEvalDataset(data.Dataset):
    """Flat-folder paired evaluation dataset.

    Input and ground-truth images are matched by filename.
    """

    def __init__(self, input_dir, gt_dir, physics_mode):
        self.input_dir = input_dir
        self.gt_dir = gt_dir
        self.physics_mode = physics_mode

        from torchvision.transforms import ToTensor

        self.to_tensor = ToTensor()
        input_files = [
            f for f in os.listdir(input_dir)
            if is_image_file(f) and os.path.isfile(os.path.join(gt_dir, f))
        ]
        self.files = sorted(input_files)

        if not self.files:
            raise RuntimeError(
                "No matched image pairs found. Make sure input and GT folders "
                "contain images with the same filenames."
            )

    def __getitem__(self, index):
        filename = self.files[index]

        inp = load_img(os.path.join(self.input_dir, filename))
        gt = load_img(os.path.join(self.gt_dir, filename))

        inp = self.to_tensor(inp)
        gt = self.to_tensor(gt)
        inp = _add_physics_channels(inp, self.physics_mode)

        return inp, gt

    def __len__(self):
        return len(self.files)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def profile_model(model, in_channels, image_size, device, repeats=50):
    dummy = torch.randn(1, in_channels, image_size, image_size, device=device)

    try:
        from thop import profile

        flops, params = profile(model, inputs=(dummy,), verbose=False)
    except Exception as exc:
        print(f"Warning: could not compute FLOPs with thop: {exc}")
        flops = None
        params = count_params(model)

    for _ in range(10):
        model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(repeats):
        model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()

    ms_per_image = (time.time() - t0) * 1000.0 / repeats
    return params, flops, ms_per_image


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained underwater enhancement checkpoint."
    )
    parser.add_argument("--model", type=str, default="unet_5ch",
                        choices=ALL_MODEL_NAMES)
    parser.add_argument("--checkpoint", "--resume", dest="checkpoint",
                        type=str, required=True,
                        help="Path to trained checkpoint .pth")
    parser.add_argument("--data_val", type=str, default="",
                        help="Input/test image folder")
    parser.add_argument("--data_valgt", type=str, default="",
                        help="Ground-truth/reference image folder")

    parser.add_argument("--batchSize", type=int, default=1)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--gpu_mode", type=str, default="true")
    parser.add_argument("--pretrained_backbone", type=str, default="false")

    parser.add_argument("--flops_size", type=int, default=256,
                        help="Square input size used for params/FLOPs profiling")
    parser.add_argument("--profile_repeats", type=int, default=50)
    parser.add_argument("--max_samples", type=int, default=0,
                        help="0 means evaluate all samples")
    parser.add_argument("--val_folder", type=str, default="./results/eval",
                        help="Folder for eval_results.json")
    return parser.parse_args()


def str2bool(value):
    return str(value).lower() in ("yes", "true", "t", "y", "1")


def main():
    args = parse_args()

    if not args.data_val:
        raise ValueError("Please provide --data_val <input_folder>")
    if not args.data_valgt:
        raise ValueError("Please provide --data_valgt <ground_truth_folder>")

    device = torch.device(
        "cuda" if str2bool(args.gpu_mode) and torch.cuda.is_available() else "cpu"
    )

    _, in_channels, physics_mode = parse_model_variant(args.model)
    model = build_model_lazy(
        args.model,
        pretrained_backbone=str2bool(args.pretrained_backbone),
    )

    if device.type == "cuda" and args.num_gpus > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(args.num_gpus)))

    model = model.to(device)
    epoch, stored_metrics = load_ckpt(
        args.checkpoint,
        model,
        optimizer=None,
        device=str(device),
    )
    model.eval()

    print("=" * 60)
    print(f"Loaded checkpoint : {args.checkpoint}")
    print(f"Model             : {args.model}")
    print(f"Epoch             : {epoch}")
    print(f"Stored metrics    : {stored_metrics}")
    print("=" * 60)

    params, flops, ms_per_image = profile_model(
        model,
        in_channels=in_channels,
        image_size=args.flops_size,
        device=device,
        repeats=args.profile_repeats,
    )

    print("\nMODEL PROFILE")
    print(f"Params : {params / 1e6:.3f} M")
    if flops is not None:
        print(f"FLOPs  : {flops / 1e9:.3f} G")
    print(f"Time   : {ms_per_image:.2f} ms/image @ {args.flops_size}x{args.flops_size}")

    dataset = PairedEvalDataset(args.data_val, args.data_valgt, physics_mode)
    loader = data.DataLoader(
        dataset,
        batch_size=args.batchSize,
        shuffle=False,
        num_workers=args.threads,
        pin_memory=(device.type == "cuda"),
    )

    max_samples = args.max_samples if args.max_samples > 0 else None
    metrics, n_images = evaluate_loader(
        model,
        loader,
        device,
        max_samples=max_samples,
        desc="Testing",
    )

    print("\nTEST RESULTS")
    print(f"Images    : {n_images}")
    print(f"PSNR      : {metrics['psnr']:.4f} dB")
    print(f"SSIM      : {metrics['ssim']:.4f}")
    print(f"CIEDE2000 : {metrics['ciede2000']:.4f}")
    print(f"UCIQE     : {metrics['uciqe']:.4f}")
    print(f"UIQM      : {metrics['uiqm']:.4f}")

    os.makedirs(args.val_folder, exist_ok=True)
    out_path = os.path.join(args.val_folder, "eval_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "model": args.model,
            "checkpoint": args.checkpoint,
            "epoch": epoch,
            "stored_metrics": stored_metrics,
            "params": params,
            "params_M": params / 1e6,
            "flops": flops,
            "flops_G": None if flops is None else flops / 1e9,
            "ms_per_image": ms_per_image,
            "num_images": n_images,
            "test_metrics": metrics,
        }, f, indent=2)
    print(f"\nSaved results: {out_path}")


if __name__ == "__main__":
    main()
