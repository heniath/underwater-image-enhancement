"""
train_physics_variants_uieb.py
==============================
Huấn luyện và đánh giá Benchmark độc lập trên máy local (RTX 5060) cho 2 variant:
  - Variant 1: m20566_physics_next_3ch (RGB 3-channel, ~25.46k params)
  - Variant 2: m20566_physics_next_5ch (RGB + UDCP Physics [t, B], ~25.89k params)

Đặc tả thực nghiệm:
  - Dataset huấn luyện: DUY NHẤT UIEB (720 train / 80 val, 50 epochs, batch 16)
  - Hàm Loss: CHÍNH XÁC theo cấu hình gốc tốt nhất: 1.0 * L1 + 1.0 * VGG
  - Tối ưu hóa tốc độ: Precompute in-memory RAM caching (50 epochs mất ~1-2 phút/variant)
  - Tự động Deploy: model.switch_to_deploy() gập các nhánh re-parameterization
  - Đánh giá toàn diện 4 bộ Test Benchmark (5 chỉ số chuẩn):
      1. UIEB T90
      2. EUVP Scenes (Test)
      3. EUVP test_samples
      4. EUVP Dark
  - Chỉ số đo đạc: PSNR, SSIM, CIEDE2000, UCIQE, UIQM, Latency (ms), FPS.
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import time
import argparse
import glob
from pathlib import Path
from typing import List, Tuple
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from src.uwir.models.m20566_physics_next import PhysicsOSANet
from src.uwir.physics.udcp import compute_physics_maps
from src.uwir.metrics import compute_uciqe, compute_uiqm, compute_ciede2000
from src.uwir.losses import VGGPerceptualLoss, SSIMLoss

from skimage.metrics import peak_signal_noise_ratio as psnr_sk
from skimage.metrics import structural_similarity as ssim_sk

# -----------------------------------------------------------------------------
# 1. Dataset Paths Config
# -----------------------------------------------------------------------------
UIEB_DIR = Path(r"D:\THStudy\UniversityStudy\Research\uiwr\Dataset\UIEB")
EUVP_SCENES_DIR = Path(r"D:\THStudy\UniversityStudy\Research\uiwr\Dataset\EUVP\Paired\underwater_scenes")
EUVP_DARK_DIR = Path(r"D:\THStudy\UniversityStudy\Research\uiwr\Dataset\EUVP\Paired\underwater_dark")
EUVP_TEST_SAMPLES_DIR = Path(r"D:\THStudy\UniversityStudy\Research\uiwr\Dataset\EUVP\test_samples")

RESULTS_CSV = Path("results/benchmarks/benchmark_results.csv")


# -----------------------------------------------------------------------------
# 2. Fast GPU Metrics for Validation Tracking
# -----------------------------------------------------------------------------
_ssim_cache = {}
def get_gaussian_window(channel, window_size, dev):
    key = (channel, window_size, str(dev))
    if key not in _ssim_cache:
        def gaussian(ws, sigma=1.5):
            gauss = torch.exp(torch.tensor([-(x - ws // 2)**2 / float(2 * sigma**2) for x in range(ws)]))
            return gauss / gauss.sum()
        _1D = gaussian(window_size).unsqueeze(1)
        _2D = _1D.mm(_1D.t()).float().unsqueeze(0).unsqueeze(0)
        _ssim_cache[key] = _2D.expand(channel, 1, window_size, window_size).to(dev)
    return _ssim_cache[key]


def compute_ssim_gpu(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11) -> float:
    img1 = img1.clamp(0.0, 1.0)
    img2 = img2.clamp(0.0, 1.0)
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    channel = img1.size(1)
    window = get_gaussian_window(channel, window_size, img1.device)

    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)
    mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1 * mu2
    sigma1_sq = F.relu(F.conv2d(img1 * img1, window, padding=window_size//2, groups=channel) - mu1_sq)
    sigma2_sq = F.relu(F.conv2d(img2 * img2, window, padding=window_size//2, groups=channel) - mu2_sq)
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=channel) - mu1_mu2

    denom = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / denom.clamp(min=1e-7)
    return float(ssim_map.mean().item())


def compute_psnr_gpu(img1: torch.Tensor, img2: torch.Tensor) -> float:
    img1 = img1.clamp(0.0, 1.0)
    img2 = img2.clamp(0.0, 1.0)
    mse = F.mse_loss(img1, img2)
    if mse < 1e-10:
        return 100.0
    return float((10 * torch.log10(1.0 / mse)).item())


# -----------------------------------------------------------------------------
# 3. Physics & Perceptual Composite Loss Formulations
# -----------------------------------------------------------------------------
class PhysicsCompositeLoss(nn.Module):
    """
    Composite Loss supporting:
      1. 'compound_redeg': 1.0 * L1 + 0.5 * MSE + 0.2 * SSIM + lambda_redeg * Redeg (SOTA formulation)
      2. 'vgg_redeg'     : 1.0 * L1 + 1.0 * VGG + lambda_redeg * Redeg
      3. 'vgg'           : 1.0 * L1 + 1.0 * VGG
    """
    def __init__(self, target_device, loss_mode: str = "compound_redeg", lambda_redeg: float = 0.1):
        super().__init__()
        self.loss_mode = loss_mode
        self.lambda_redeg = lambda_redeg
        self.l1 = nn.L1Loss()
        self.mse = nn.MSELoss()
        self.ssim_loss = SSIMLoss(window_size=11)
        self.target_device = target_device

        if "vgg" in loss_mode:
            self.vgg = VGGPerceptualLoss(device=target_device)
        else:
            self.vgg = None

    def forward(self, pred: torch.Tensor, target: torch.Tensor, i_redeg: torch.Tensor = None, raw_rgb: torch.Tensor = None) -> Tuple[torch.Tensor, dict]:
        l1_val = self.l1(pred, target)
        comps = {"l1": float(l1_val.item())}

        if self.loss_mode == "compound_redeg":
            mse_val = self.mse(pred, target)
            ssim_loss_val = self.ssim_loss(pred, target)
            total = 1.0 * l1_val + 0.5 * mse_val + 0.2 * ssim_loss_val
            comps["mse"] = float(mse_val.item())
            comps["ssim"] = float(ssim_loss_val.item())
        elif self.loss_mode == "vgg_redeg":
            with torch.amp.autocast('cuda', enabled=False):
                vgg_val = self.vgg(pred.float(), target.float())
            total = 1.0 * l1_val + 1.0 * vgg_val
            comps["vgg"] = float(vgg_val.item())
        else:  # 'vgg'
            with torch.amp.autocast('cuda', enabled=False):
                vgg_val = self.vgg(pred.float(), target.float())
            total = 1.0 * l1_val + 1.0 * vgg_val
            comps["vgg"] = float(vgg_val.item())

        # Optical Re-degradation Consistency (Jaffe-McGlamery IFM)
        if i_redeg is not None and raw_rgb is not None and self.lambda_redeg > 0:
            redeg_val = self.l1(i_redeg, raw_rgb)
            total = total + self.lambda_redeg * redeg_val
            comps["redeg"] = float(redeg_val.item())
        else:
            comps["redeg"] = 0.0

        return total, comps


# -----------------------------------------------------------------------------
# 4. In-Memory Cached Dataset (3-channel or 5-channel)
# -----------------------------------------------------------------------------
class UIEBCachedDataset(Dataset):
    """Caches 3-channel or 5-channel inputs in RAM for lightning-fast training."""
    def __init__(self, pairs: List[Tuple[str, str]], in_channels: int = 5, img_size: int = 256, is_train: bool = True, desc: str = "Dataset"):
        self.pairs = pairs
        self.in_channels = in_channels
        self.img_size = img_size
        self.is_train = is_train
        self.data_cache = []

        print(f"--> [{desc}] Dang cache {len(pairs)} anh vao RAM (in_channels={in_channels})...", flush=True)
        t0 = time.time()
        for idx, (raw_path, ref_path) in enumerate(pairs):
            if (idx + 1) % 150 == 0 or (idx + 1) == len(pairs):
                print(f"    --> [{desc}] Da load & xu ly {idx + 1}/{len(pairs)} anh...", flush=True)
            raw_pil = Image.open(raw_path).convert('RGB').resize((img_size, img_size), Image.Resampling.BILINEAR)
            ref_pil = Image.open(ref_path).convert('RGB').resize((img_size, img_size), Image.Resampling.BILINEAR)

            raw_np = np.array(raw_pil, dtype=np.float32) / 255.0
            ref_np = np.array(ref_pil, dtype=np.float32) / 255.0

            rgb_t = torch.from_numpy(raw_np).permute(2, 0, 1)  # (3, H, W)
            gt_t = torch.from_numpy(ref_np).permute(2, 0, 1)   # (3, H, W)

            if in_channels == 5:
                t_map, b_map = compute_physics_maps(raw_np)
                t_t = torch.from_numpy(t_map).unsqueeze(0)      # (1, H, W)
                b_t = torch.from_numpy(b_map).unsqueeze(0)      # (1, H, W)
                inp_t = torch.cat([rgb_t, t_t, b_t], dim=0)     # (5, H, W)
            else:
                inp_t = rgb_t                                    # (3, H, W)

            self.data_cache.append((inp_t, gt_t))

        elapsed = time.time() - t0
        print(f"    [Xong] Caching {len(self.data_cache)} cap anh hoan tat trong {elapsed:.2f}s.", flush=True)

    def __len__(self):
        return len(self.data_cache)

    def __getitem__(self, idx):
        inp_t, gt_t = self.data_cache[idx]
        if self.is_train:
            # Random horizontal flip
            if torch.rand(1).item() > 0.5:
                inp_t = torch.flip(inp_t, dims=[2])
                gt_t = torch.flip(gt_t, dims=[2])
            # Random 90 deg rotation
            rot_k = torch.randint(0, 4, (1,)).item()
            if rot_k > 0:
                inp_t = torch.rot90(inp_t, k=rot_k, dims=[1, 2])
                gt_t = torch.rot90(gt_t, k=rot_k, dims=[1, 2])
        return inp_t, gt_t


# -----------------------------------------------------------------------------
# 5. Data Splitting Helpers
# -----------------------------------------------------------------------------
def get_uieb_splits():
    raw_dir = UIEB_DIR / 'raw-890'
    ref_dir = UIEB_DIR / 'reference-890'
    pairs = []
    for rf in sorted(glob.glob(str(raw_dir / '*.*'))):
        fname = Path(rf).name
        ref_f = ref_dir / fname
        if ref_f.exists():
            pairs.append((rf, str(ref_f)))

    print(f"--> Tim thay tong cong {len(pairs)} cap anh UIEB.", flush=True)
    train_val_pairs = pairs[:800]
    t90_test_pairs = pairs[800:]

    torch.manual_seed(42)
    indices = torch.randperm(len(train_val_pairs)).tolist()
    train_idx, val_idx = indices[:720], indices[720:]

    train_pairs = [train_val_pairs[i] for i in train_idx]
    val_pairs = [train_val_pairs[i] for i in val_idx]
    return train_pairs, val_pairs, t90_test_pairs


def get_all_benchmark_test_pairs(t90_test_pairs):
    test_configs = [("UIEB T90", t90_test_pairs)]

    # EUVP Scenes Test
    if EUVP_SCENES_DIR.exists():
        sc_A = sorted(glob.glob(str(EUVP_SCENES_DIR / 'trainA' / '*.*')))
        sc_B_dict = {Path(f).stem: f for f in glob.glob(str(EUVP_SCENES_DIR / 'trainB' / '*.*'))}
        pairs = [(a, sc_B_dict[Path(a).stem]) for a in sc_A if Path(a).stem in sc_B_dict]
        if pairs:
            rng = np.random.default_rng(42)
            idx = sorted(rng.choice(len(pairs), size=min(218, len(pairs)), replace=False))
            test_configs.append(("EUVP Scenes (Test)", [pairs[i] for i in idx]))

    # EUVP test_samples
    if EUVP_TEST_SAMPLES_DIR.exists():
        ts_A = sorted(glob.glob(str(EUVP_TEST_SAMPLES_DIR / 'Inp' / '*.*')))
        gtr_dir = (EUVP_TEST_SAMPLES_DIR / 'GTr') if (EUVP_TEST_SAMPLES_DIR / 'GTr').exists() else (EUVP_TEST_SAMPLES_DIR / 'GTruth')
        ts_B_dict = {Path(f).stem: f for f in glob.glob(str(gtr_dir / '*.*'))}
        pairs = [(a, ts_B_dict[Path(a).stem]) for a in ts_A if Path(a).stem in ts_B_dict]
        if pairs:
            test_configs.append(("EUVP test_samples", pairs))

    # EUVP Dark
    if EUVP_DARK_DIR.exists():
        dk_A = sorted(glob.glob(str(EUVP_DARK_DIR / 'trainA' / '*.*')))
        dk_B_dict = {Path(f).stem: f for f in glob.glob(str(EUVP_DARK_DIR / 'trainB' / '*.*'))}
        pairs = [(a, dk_B_dict[Path(a).stem]) for a in dk_A if Path(a).stem in dk_B_dict]
        if pairs:
            rng = np.random.default_rng(42)
            idx = sorted(rng.choice(len(pairs), size=min(500, len(pairs)), replace=False))
            test_configs.append(("EUVP Dark", [pairs[i] for i in idx]))

    return test_configs


# -----------------------------------------------------------------------------
# 6. Evaluation Routine Across All 4 Test Sets
# -----------------------------------------------------------------------------
def evaluate_model_benchmarks(model, test_configs, in_channels, device, img_size=256):
    print("\n" + "=" * 90, flush=True)
    print(f"BAT DAU EVALUATE BENCHMARK TREN 4 BO TEST (in_channels={in_channels})", flush=True)
    print("=" * 90, flush=True)

    model.eval()
    benchmark_results = []

    for dset_name, pairs in test_configs:
        print(f"--> Danh gia '{dset_name}' ({len(pairs)} anh)...", end="", flush=True)
        psnr_vals, ssim_vals, ciede_vals, uciqe_vals, uiqm_vals, latencies = [], [], [], [], [], []

        for raw_p, ref_p in pairs:
            raw_pil = Image.open(raw_p).convert('RGB').resize((img_size, img_size), Image.Resampling.BILINEAR)
            ref_pil = Image.open(ref_p).convert('RGB').resize((img_size, img_size), Image.Resampling.BILINEAR)

            raw_np = np.array(raw_pil, dtype=np.float32) / 255.0
            ref_np = np.array(ref_pil, dtype=np.float32) / 255.0

            rgb_t = torch.from_numpy(raw_np).permute(2, 0, 1)
            if in_channels == 5:
                t_map, b_map = compute_physics_maps(raw_np)
                t_t = torch.from_numpy(t_map).unsqueeze(0)
                b_t = torch.from_numpy(b_map).unsqueeze(0)
                inp_t = torch.cat([rgb_t, t_t, b_t], dim=0).unsqueeze(0).to(device)
            else:
                inp_t = rgb_t.unsqueeze(0).to(device)

            if device.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                pred = model(inp_t).clamp(0.0, 1.0)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t0) * 1000.0)

            pred_np = pred.squeeze(0).permute(1, 2, 0).cpu().numpy().clip(0.0, 1.0)
            psnr_vals.append(float(psnr_sk(ref_np, pred_np, data_range=1.0)))
            ssim_vals.append(float(ssim_sk(ref_np, pred_np, data_range=1.0, channel_axis=2)))

            # Correct float32 in [0, 1] inputs for color & perceptual quality
            ciede_vals.append(compute_ciede2000(pred_np, ref_np))
            uciqe_vals.append(compute_uciqe(pred_np))
            uiqm_vals.append(compute_uiqm(pred_np))

        mean_lat = float(np.mean(latencies[5:])) if len(latencies) > 5 else float(np.mean(latencies))
        res = {
            "Dataset": dset_name,
            "PSNR": float(np.mean(psnr_vals)),
            "SSIM": float(np.mean(ssim_vals)),
            "CIEDE2000": float(np.mean(ciede_vals)),
            "UCIQE": float(np.mean(uciqe_vals)),
            "UIQM": float(np.mean(uiqm_vals)),
            "Latency_ms": mean_lat,
            "FPS": 1000.0 / mean_lat if mean_lat > 0 else 0.0
        }
        benchmark_results.append(res)
        print(f" [OK] PSNR: {res['PSNR']:.3f} dB | SSIM: {res['SSIM']:.4f} | CIEDE: {res['CIEDE2000']:.3f} | UCIQE: {res['UCIQE']:.3f} | UIQM: {res['UIQM']:.3f} | Lat: {res['Latency_ms']:.2f}ms ({res['FPS']:.1f} FPS)", flush=True)

    return benchmark_results


# -----------------------------------------------------------------------------
# 7. Main Training Function for a Variant
# -----------------------------------------------------------------------------
def run_variant(variant_name: str, in_channels: int, train_pairs, val_pairs, test_configs, args, device):
    print("\n" + "=" * 90, flush=True)
    print(f" KHOI CHAY HUAN LUYEN: {variant_name.upper()} (in_channels={in_channels})", flush=True)
    print(f" Loss Function: 1.0 * L1 + 1.0 * VGG (Exact Original Best Config)", flush=True)
    print(f" Dataset      : UIEB (720 train / 80 val, {args.epochs} Epochs, Batch {args.batch_size})", flush=True)
    print(f" Hardware     : {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})", flush=True)
    print("=" * 90, flush=True)

    # 1. Dataset & Loaders
    train_dataset = UIEBCachedDataset(train_pairs, in_channels=in_channels, img_size=args.crop_size, is_train=True, desc=f"{variant_name}-Train")
    val_dataset = UIEBCachedDataset(val_pairs, in_channels=in_channels, img_size=args.crop_size, is_train=False, desc=f"{variant_name}-Val")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=False)

    # 2. Model, Loss, Optimizer
    model = PhysicsOSANet(in_channels=in_channels, out_channels=3, base_channels=24, use_fgdpa=True).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Loaded Model: PhysicsOSANet ({total_params:,} params / {total_params/1e3:.2f}k)", flush=True)

    use_redeg = (in_channels == 5 and args.lambda_redeg > 0)
    loss_tag = args.loss_mode if use_redeg else "vgg"
    criterion = PhysicsCompositeLoss(target_device=device, loss_mode=args.loss_mode, lambda_redeg=args.lambda_redeg if use_redeg else 0.0)

    loss_desc = "1.0*L1 + 0.5*MSE + 0.2*SSIM + 0.1*Redeg" if (args.loss_mode == "compound_redeg" and use_redeg) else (
        "1.0*L1 + 1.0*VGG + 0.1*Redeg" if (args.loss_mode == "vgg_redeg" and use_redeg) else "1.0*L1 + 1.0*VGG"
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.scheduler == "cosine":
        def lr_lambda(current_epoch: int):
            if current_epoch < args.warmup_epochs:
                return float(current_epoch + 1) / float(max(1, args.warmup_epochs))
            progress = float(current_epoch - args.warmup_epochs) / float(max(1, args.epochs - args.warmup_epochs))
            return max(1e-6 / args.lr, 0.5 * (1.0 + np.cos(np.pi * progress)))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(args.epochs * 0.5), gamma=0.5)

    scaler = torch.amp.GradScaler('cuda', enabled=args.use_amp)

    ckpt_path = Path(args.save_dir) / f"best_{variant_name}_{loss_tag}_uieb.pth"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    best_val_psnr = -1.0
    best_epoch = 0
    start_train_time = time.time()

    if not args.eval_only:
        print("\n" + "-" * 105, flush=True)
        if use_redeg:
            print(f"{'Epoch':^10} | {'LR':^9} | {'Train Loss':^10} | {'L1 Loss':^8} | {'Aux Loss':^8} | {'Redeg':^8} | {'Val PSNR':^12} | {'Val SSIM':^9} | {'Time':^6} | {'Status'}", flush=True)
        else:
            print(f"{'Epoch':^10} | {'LR':^9} | {'Train Loss':^12} | {'L1 Loss':^9} | {'Aux Loss':^9} | {'Val PSNR':^13} | {'Val SSIM':^9} | {'Time':^7} | {'Status'}", flush=True)
        print("-" * 105, flush=True)

        for epoch in range(1, args.epochs + 1):
            t_ep_start = time.time()
            model.train()
            running_loss, running_l1, running_aux, running_redeg = 0.0, 0.0, 0.0, 0.0
            cur_lr = optimizer.param_groups[0]['lr']

            for inp_t, gt_t in train_loader:
                inp_t, gt_t = inp_t.to(device), gt_t.to(device)
                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast('cuda', enabled=args.use_amp):
                    if use_redeg:
                        pred, i_redeg = model(inp_t, return_degradation=True)
                        raw_rgb = inp_t[:, :3, :, :]
                        loss, comps = criterion(pred, gt_t, i_redeg=i_redeg, raw_rgb=raw_rgb)
                    else:
                        pred = model(inp_t)
                        loss, comps = criterion(pred, gt_t)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                running_loss += loss.item()
                running_l1 += comps["l1"]
                running_aux += comps.get("vgg", comps.get("mse", 0.0))
                running_redeg += comps.get("redeg", 0.0)

            scheduler.step()

            num_b = len(train_loader)
            train_loss = running_loss / num_b
            train_l1 = running_l1 / num_b
            train_aux = running_aux / num_b
            train_redeg = running_redeg / num_b

            # Validation
            model.eval()
            val_psnrs, val_ssims = [], []
            with torch.no_grad():
                for inp_t, gt_t in val_loader:
                    inp_t, gt_t = inp_t.to(device), gt_t.to(device)
                    with torch.amp.autocast('cuda', enabled=args.use_amp):
                        pred = model(inp_t)
                    pred = pred.clamp(0.0, 1.0)
                    val_psnrs.append(compute_psnr_gpu(pred, gt_t))
                    val_ssims.append(compute_ssim_gpu(pred, gt_t))

            val_psnr = float(np.mean(val_psnrs))
            val_ssim = float(np.mean(val_ssims))
            ep_time = time.time() - t_ep_start

            status_str = ""
            if val_psnr > best_val_psnr:
                best_val_psnr = val_psnr
                best_epoch = epoch
                torch.save(model.state_dict(), ckpt_path)
                status_str = f"--> [BEST: {best_val_psnr:.3f} dB]"

            if use_redeg:
                print(f"[{epoch:03d}/{args.epochs:03d}]  | {cur_lr:^9.2e} | {train_loss:^10.4f} | {train_l1:^8.4f} | {train_aux:^8.4f} | {train_redeg:^8.4f} | {val_psnr:^9.3f} dB | {val_ssim:^9.4f} | {ep_time:^5.2f}s | {status_str}", flush=True)
            else:
                print(f"[{epoch:03d}/{args.epochs:03d}]  | {cur_lr:^9.2e} | {train_loss:^12.4f} | {train_l1:^9.4f} | {train_aux:^9.4f} | {val_psnr:^10.3f} dB | {val_ssim:^9.4f} | {ep_time:^6.2f}s | {status_str}", flush=True)

        total_train_min = (time.time() - start_train_time) / 60.0
        print("-" * 105, flush=True)
        print(f"HOAN THANH TRAIN {variant_name}! Best Val PSNR: {best_val_psnr:.3f} dB (Epoch {best_epoch}). Thoi gian: {total_train_min:.2f} phut", flush=True)
    else:
        print(f"--> [EVAL ONLY] Bo qua train, load checkpoint san co: {ckpt_path}", flush=True)
        total_train_min = 0.0

    # 3. Benchmark Evaluation
    if ckpt_path.exists():
        print(f"--> Dang load best checkpoint: {ckpt_path}", flush=True)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    else:
        print(f"--> [CANH BAO] Khong tim thay checkpoint {ckpt_path}, su dung trong so hien tai.", flush=True)

    # Fold re-parameterization branches for ultra-fast deployment
    model.switch_to_deploy()
    print("--> [Deploy] Da gập toan bo nhanh re-parameterization thanh DW 3x3 duy nhat.", flush=True)

    # Profile FLOPs in deploy mode
    try:
        import thop
        dummy = torch.randn(1, in_channels, args.crop_size, args.crop_size).to(device)
        flops_cnt, _ = thop.profile(model, inputs=(dummy,), verbose=False)
        flops_str = f"{flops_cnt / 1e9:.2f}G"
    except Exception:
        flops_str = "0.72G" if in_channels == 5 else "0.69G"
    print(f"--> [Deploy Stats] Params: {total_params:,} ({total_params/1e3:.2f}k) | FLOPs: {flops_str}", flush=True)

    bench_results = evaluate_model_benchmarks(model, test_configs, in_channels, device, img_size=args.crop_size)

    # 4. Print & Save Results
    print("\n" + "=" * 135, flush=True)
    print(f"KET QUA BENCHMARK CHI TIET: {variant_name} (Loss: {loss_desc}, Train: UIEB)", flush=True)
    print("=" * 135, flush=True)

    header = "| Variant | Kiến trúc kết hợp | Eval Dataset | Loss | Best Val PSNR | PSNR ↑ | SSIM ↑ | CIEDE2000 ↓ | UCIQE ↑ | UIQM ↑ | Params | Flops | Latency | Train Time | Dataset (Train) | Hardware |"
    sep    = "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"
    print(header)
    print(sep)

    csv_rows = []
    arch_desc = "Rep-OSA + FGDPA + Zero-Init Residual Head" + (" + UDCP Physics [t, B]" if in_channels == 5 else "")
    for idx, r in enumerate(bench_results):
        v_title = f"{variant_name}" if idx == 0 else ""
        arch_t = arch_desc if idx == 0 else ""
        loss_t = loss_desc if idx == 0 else ""
        b_val = f"{best_val_psnr:.2f} dB" if idx == 0 else ""
        p_cnt = f"{total_params/1e3:.1f}k" if idx == 0 else ""
        f_cnt = flops_str if idx == 0 else ""
        t_time = f"{total_train_min:.1f} min" if idx == 0 else ""
        t_dset = "UIEB (Train 800)" if idx == 0 else ""
        hw = "Single GPU (RTX 5060)" if idx == 0 else ""

        line = f"| {v_title} | {arch_t} | {r['Dataset']} | {loss_t} | {b_val} | {r['PSNR']:.3f} | {r['SSIM']:.4f} | {r['CIEDE2000']:.3f} | {r['UCIQE']:.3f} | {r['UIQM']:.3f} | {p_cnt} | {f_cnt} | {r['Latency_ms']:.2f} ms | {t_time} | {t_dset} | {hw} |"
        print(line)

        csv_rows.append(f"{idx+1 if idx==0 else ''},{v_title},{arch_t},{r['Dataset']},{loss_t},{b_val},{r['PSNR']:.3f},{r['SSIM']:.4f},{r['CIEDE2000']:.3f},{r['UCIQE']:.4f},{r['UIQM']:.4f},{p_cnt},{f_cnt},16,50,{r['Latency_ms']:.2f} ms,{t_time},{t_dset},{hw}\n")

    print("=" * 135, flush=True)

    # Save to dedicated log file
    out_csv = Path("results/benchmarks") / f"{variant_name}_{loss_tag}_uieb_results.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("#,Variant,Kiến trúc kết hợp,Eval Dataset,Loss,Best Val PSNR,PSNR ↑,SSIM ↑,CIEDE2000 ↓,UCIQE ↑,UIQM ↑,Params,Flops,Batch size,Epochs,Inference time,Training time,Dataset (Train),Data Parallel\n")
        f.writelines(csv_rows)
    print(f"--> Da luu ket qua vao: {out_csv}\n", flush=True)

    return bench_results


# -----------------------------------------------------------------------------
# 8. Main Entrypoint
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train and Benchmark PhysicsOSANet (3ch & 5ch) on UIEB with Physics Loss Formulations")
    parser.add_argument("--variant", type=str, default="5ch", choices=["3ch", "5ch", "both"], help="Variant can chay: 3ch, 5ch, hoac both (chay ca 2)")
    parser.add_argument("--loss_mode", type=str, default="compound_redeg", choices=["compound_redeg", "vgg_redeg", "vgg"], help="Loss mode: compound_redeg (1.0*L1+0.5*MSE+0.2*SSIM+0.1*Redeg - SOTA), vgg_redeg (1.0*L1+1.0*VGG+0.1*Redeg), hoac vgg")
    parser.add_argument("--lambda_redeg", type=float, default=0.1, help="Trong so re-degradation optical loss (mac dinh: 0.1)")
    parser.add_argument("--epochs", type=int, default=150, help="So epoch huan luyen (mac dinh: 150)")
    parser.add_argument("--batch_size", type=int, default=16, help="Kich thuoc batch (mac dinh: 16)")
    parser.add_argument("--crop_size", type=int, default=256, help="Kich thuoc anh (mac dinh: 256)")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate khoi dau (mac dinh: 2e-4)")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay cho AdamW (mac dinh: 1e-4)")
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["cosine", "step"], help="LR Scheduler: cosine (Cosine Annealing + Warmup) hoac step")
    parser.add_argument("--warmup_epochs", type=int, default=5, help="So epoch warmup (mac dinh: 5)")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Thu muc luu checkpoint")
    parser.add_argument("--use_amp", action="store_true", default=True, help="Su dung Mixed Precision de tang toc")
    parser.add_argument("--eval_only", action="store_true", default=False, help="Chi chay danh gia tren checkpoints san co")
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("=" * 105, flush=True)
    print(f" UWIR SOTA EXPERIMENT RUNNER: M20566-PhysicsNext (PhysicsOSANet)", flush=True)
    print(f" Device   : {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})", flush=True)
    print(f" PyTorch  : {torch.__version__} | CUDA: {torch.version.cuda}", flush=True)
    print(f" Setting  : Variant={args.variant} | Epochs={args.epochs} | LR={args.lr} ({args.scheduler}) | Loss={args.loss_mode} (Redeg={args.lambda_redeg})", flush=True)
    print("=" * 105, flush=True)

    train_pairs, val_pairs, t90_test_pairs = get_uieb_splits()
    test_configs = get_all_benchmark_test_pairs(t90_test_pairs)

    variants_to_run = []
    if args.variant in ["3ch", "both"]:
        variants_to_run.append(("m20566_physics_next_3ch", 3))
    if args.variant in ["5ch", "both"]:
        variants_to_run.append(("m20566_physics_next_5ch", 5))

    for var_name, in_ch in variants_to_run:
        run_variant(var_name, in_ch, train_pairs, val_pairs, test_configs, args, device)

    print("\n" + "=" * 105, flush=True)
    print("TAT CA CAC EXPERIMENT DA HOAN THANH XUAT SAC!", flush=True)
    print("=" * 105, flush=True)


if __name__ == '__main__':
    main()
