# ============================================================
# train.py  –  Training loop
# ============================================================

import copy
import json
import os
import random
import sys
import time
from datetime import datetime
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data

from uwir.config import option
from uwir.models import build_model, parse_model_variant
from uwir.physics import PhysicsConfig, compute_physics_features
from uwir.training.schedulers import (
    CosineAnnealingRestartCyclicLR,
    CosineAnnealingRestartLR,
    GradualWarmupScheduler,
)

# ============================================================
# Tee: mirror stdout to both terminal and a log file
# ============================================================


class _Tee:
    """Write to *stream* and *file_obj* simultaneously."""

    def __init__(self, stream, file_obj):
        self._stream = stream
        self._file = file_obj

    def write(self, data: str):
        self._stream.write(data)
        self._file.write(data)

    def flush(self):
        self._stream.flush()
        self._file.flush()

    def __getattr__(self, name):
        return getattr(self._stream, name)


# ============================================================
# Physics pre-processing collate helper
# ============================================================


def _resolve_physics_extractor(prior_method: str):
    """Select the physics feature extractor requested by --prior_method."""
    from uwir.physics import (
        compute_physics_maps,
        compute_physics_maps_gdcp,
        compute_physics_maps_gupdm,
    )

    if prior_method == "udcp":
        return compute_physics_maps
    if prior_method == "gdcp":
        return compute_physics_maps_gdcp
    if prior_method == "gupdm":
        return compute_physics_maps_gupdm
    if prior_method == "multi_prior":
        raise ValueError(
            "--prior_method multi_prior is not wired to a single 4/5-channel "
            "model input. Use udcp, gdcp, or gupdm."
        )
    raise ValueError(f"Unknown --prior_method: {prior_method}")


def _add_physics_channels(
    rgb_tensor: torch.Tensor,
    mode: str,
    physics_extractor=None,
    fusion_extractor=None,
) -> torch.Tensor:
    """
    Append physics-derived channels to an RGB tensor.

    Args:
        rgb_tensor (Tensor): (3, H, W) float32 in [0, 1].
        mode       (str):    Which channels to append:
                               ``"none"`` → return as-is           (3-ch)
                               ``"t"``    → append t(x)            (4-ch)
                               ``"b"``    → append B_map           (4-ch)
                               ``"tb"``   → append t(x) and B_map  (5-ch)

    Returns:
        Tensor: (C_out, H, W) where C_out ∈ {3, 4, 5}.
    """
    if mode == "none":
        return rgb_tensor

    if mode == "tv":
        if fusion_extractor is None:
            from uwir.physics import compute_physics_features

            fusion_extractor = compute_physics_features
        image = rgb_tensor.permute(1, 2, 0).numpy().astype(np.float32)
        features = fusion_extractor(image)
        maps = torch.from_numpy(features.as_7ch_maps()).permute(2, 0, 1)
        return torch.cat([rgb_tensor, maps], dim=0)

    if physics_extractor is None:
        from uwir.physics import compute_physics_maps

        physics_extractor = compute_physics_maps

    img_np = rgb_tensor.permute(1, 2, 0).numpy().astype(np.float32)
    t_map, b_map = physics_extractor(img_np)

    if mode == "t":
        t_t = torch.from_numpy(t_map).unsqueeze(0)  # (1, H, W)
        return torch.cat([rgb_tensor, t_t], dim=0)  # (4, H, W)

    if mode == "b":
        b_t = torch.from_numpy(b_map).unsqueeze(0)  # (1, H, W)
        return torch.cat([rgb_tensor, b_t], dim=0)  # (4, H, W)

    if mode == "tb":
        t_t = torch.from_numpy(t_map).unsqueeze(0)  # (1, H, W)
        b_t = torch.from_numpy(b_map).unsqueeze(0)  # (1, H, W)
        return torch.cat([rgb_tensor, t_t, b_t], dim=0)  # (5, H, W)

    raise ValueError(f"Unknown physics mode: '{mode}'")


def _collate_train(batch, physics_mode: str, physics_extractor=None, fusion_extractor=None):
    """
    Custom collate that:
      - Drops the filename strings returned by the dataset.
      - Appends physics channels according to ``physics_mode``.

    Dataset items are (inp_tensor, gt_tensor, fname_in, fname_gt).
    """
    inps = []
    gts = []
    for inp, gt, *_ in batch:
        inp = _add_physics_channels(inp, physics_mode, physics_extractor, fusion_extractor)
        inps.append(inp)
        gts.append(gt)
    return torch.stack(inps), torch.stack(gts)


def _collate_val(batch, physics_mode: str, physics_extractor=None, fusion_extractor=None):
    """Same as _collate_train but for validation paired datasets."""
    return _collate_train(batch, physics_mode, physics_extractor, fusion_extractor)


def _dataloader_kwargs(args, device: torch.device) -> dict:
    kwargs = {
        "num_workers": args.threads,
        "pin_memory": device.type == "cuda",
    }
    if args.threads > 0:
        kwargs["persistent_workers"] = args.persistent_workers
        kwargs["prefetch_factor"] = max(1, args.prefetch_factor)
    return kwargs


def _validation_copy(dataset):
    """Shallow-copy a dataset tree with stochastic augmentation disabled."""
    if isinstance(dataset, data.ConcatDataset):
        return data.ConcatDataset([_validation_copy(child) for child in dataset.datasets])
    cloned = copy.copy(dataset)
    if hasattr(cloned, "augment"):
        cloned.augment = False
    return cloned


def _split_train_validation(dataset, seed: int, fraction: float = 0.10):
    """Return train/validation subsets with a deterministic stratified EUVP split."""
    generator = torch.Generator().manual_seed(seed)
    groups: dict[str, list[int]] = {}
    if hasattr(dataset, "input_files"):
        for index, filename in enumerate(dataset.input_files):
            path = Path(filename)
            group = path.parent.parent.name if path.parent.name == "trainA" else "all"
            groups.setdefault(group, []).append(index)
    else:
        groups["all"] = list(range(len(dataset)))

    train_indices: list[int] = []
    val_indices: list[int] = []
    for indices in groups.values():
        order = torch.randperm(len(indices), generator=generator).tolist()
        n_val = max(1, int(len(indices) * fraction))
        val_indices.extend(indices[position] for position in order[:n_val])
        train_indices.extend(indices[position] for position in order[n_val:])
    train_indices.sort()
    val_indices.sort()
    return (
        data.Subset(dataset, train_indices),
        data.Subset(_validation_copy(dataset), val_indices),
    )


def _subset_manifest(subset) -> list[str]:
    """Return stable input paths for a Subset when its base exposes them."""
    base = subset.dataset
    if hasattr(base, "input_files"):
        return [str(base.input_files[index]) for index in subset.indices]
    return [str(index) for index in subset.indices]


# ============================================================
# EarlyStopping
# ============================================================


class EarlyStopping:
    def __init__(self, patience=15, min_delta=1e-4, mode="max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best = None
        self.stop = False

    def __call__(self, score):
        if self.best is None:
            self.best = score
        elif (self.mode == "max" and score > self.best + self.min_delta) or (
            self.mode == "min" and score < self.best - self.min_delta
        ):
            self.best = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
        return self.stop


# ============================================================
# Checkpoint helpers
# ============================================================


def _unwrap(model):
    """Return the bare nn.Module, stripping DataParallel if present."""
    return model.module if isinstance(model, nn.DataParallel) else model


def save_ckpt(model, optimizer, epoch, metrics, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": _unwrap(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "metrics": metrics,
        },
        path,
    )


def load_ckpt(path, model, optimizer=None, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    _unwrap(model).load_state_dict(ckpt["model"])
    if optimizer and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt["epoch"], ckpt.get("metrics", {})


# ============================================================
# Inner-loop functions
# ============================================================


def _last_prediction(pred):
    if isinstance(pred, (list, tuple)):
        if not pred:
            raise ValueError("Model returned an empty prediction list.")
        return pred[-1]
    return pred


def _criterion_with_deep_supervision(criterion, pred, gt):
    if not isinstance(pred, (list, tuple)):
        return criterion(pred, gt)

    losses = []
    part_sums = {}
    base_weights = (0.1, 0.2, 0.4, 1.0)
    if len(pred) <= len(base_weights):
        weights = base_weights[-len(pred) :]
    else:
        weights = (base_weights[0],) * (len(pred) - len(base_weights)) + base_weights

    for output, weight in zip(pred, weights):
        loss, parts = criterion(output, gt)
        losses.append(loss * weight)
        for key, value in parts.items():
            part_sums[key] = part_sums.get(key, 0.0) + (value * weight)

    if not losses:
        raise ValueError("Model returned an empty prediction list.")

    weight_sum = float(sum(weights))
    total = torch.stack(losses).sum() / weight_sum
    avg_parts = {key: value / weight_sum for key, value in part_sums.items()}
    avg_parts["total"] = total.item()
    return total, avg_parts


def train_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    scaler=None,
    accumulation_steps: int = 1,
    log_batch_timing: bool = False,
):
    model.train()
    tot_loss = 0.0
    comps = {"l1": 0.0, "perceptual": 0.0, "ssim_loss": 0.0}
    consecutive_amp_overflows = 0
    data_wait_acc = 0.0
    step_acc = 0.0
    last_step_end = time.time()

    # BỎ TQDM, DÙNG ENUMERATE THÔNG THƯỜNG
    for batch_idx, (inp, gt) in enumerate(loader):
        batch_ready = time.time()
        if log_batch_timing:
            data_wait_acc += batch_ready - last_step_end
        inp = inp.to(device, non_blocking=True)
        gt = gt.to(device, non_blocking=True)
        if batch_idx % accumulation_steps == 0:
            optimizer.zero_grad(set_to_none=True)
        amp_enabled = scaler is not None and scaler.is_enabled()
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            pred = model(inp)
            loss, parts = _criterion_with_deep_supervision(criterion, pred, gt)
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite training loss at batch {batch_idx + 1}: {loss.item()}"
            )
        scaled_loss = loss / accumulation_steps
        if scaler is not None:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()
        should_step = (batch_idx + 1) % accumulation_steps == 0 or batch_idx + 1 == len(loader)
        if should_step:
            if scaler is not None:
                scaler.unscale_(optimizer)
            try:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            except RuntimeError as exc:
                if not amp_enabled:
                    raise FloatingPointError(
                        f"Non-finite gradient norm at batch {batch_idx + 1}"
                    ) from exc

                # Overflow is expected occasionally with dynamic loss
                # scaling. Skip this optimizer step and reduce the scale;
                # raising immediately prevents GradScaler from doing its job.
                consecutive_amp_overflows += 1
                old_scale = scaler.get_scale()
                bad_parameters = (
                    [
                        name
                        for name, parameter in model.named_parameters()
                        if parameter.grad is not None and not torch.isfinite(parameter.grad).all()
                    ]
                    if consecutive_amp_overflows >= 8
                    else []
                )
                scaler.update(new_scale=max(old_scale / 2.0, 1.0))
                optimizer.zero_grad(set_to_none=True)
                print(
                    f"   [AMP] skipped batch {batch_idx + 1} after gradient overflow; "
                    f"scale {old_scale:g} -> {scaler.get_scale():g}"
                )
                if consecutive_amp_overflows >= 8:
                    names = ", ".join(bad_parameters[:12]) or "unknown"
                    raise FloatingPointError(
                        "Persistent non-finite AMP gradients after 8 scale reductions "
                        f"at batch {batch_idx + 1}; parameters: {names}"
                    ) from exc
            else:
                consecutive_amp_overflows = 0
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

        tot_loss += loss.item()
        for k in comps:
            comps[k] += parts.get(k, 0.0)

        # Chỉ in log ra màn hình mỗi 50 batch để tránh tràn I/O
        if (batch_idx + 1) % 50 == 0:
            if log_batch_timing and device.type == "cuda":
                torch.cuda.synchronize()
            step_end = time.time()
            if log_batch_timing:
                step_acc += step_end - batch_ready
                window = 50
                print(
                    f"   [Batch {batch_idx + 1}/{len(loader)}] "
                    f"Loss: {loss.item():.4f} | "
                    f"data_wait: {data_wait_acc / window:.3f}s/b | "
                    f"step: {step_acc / window:.3f}s/b"
                )
                data_wait_acc = 0.0
                step_acc = 0.0
            else:
                print(f"   [Batch {batch_idx + 1}/{len(loader)}] Loss: {loss.item():.4f}")
            last_step_end = step_end
        else:
            if log_batch_timing:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                step_end = time.time()
                step_acc += step_end - batch_ready
                last_step_end = step_end

    n = len(loader)
    return tot_loss / n, {k: v / n for k, v in comps.items()}


@torch.no_grad()
def val_loss_epoch(model, loader, criterion, device, amp_enabled: bool = False):
    model.eval()
    tot = 0.0
    for batch_idx, (inp, gt) in enumerate(loader):
        inp = inp.to(device, non_blocking=True)
        gt = gt.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            pred = model(inp)
            pred = _last_prediction(pred)
            loss, _ = criterion(pred, gt)
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite validation loss at batch {batch_idx + 1}: {loss.item()}"
            )
        tot += loss.item()
    return tot / len(loader)


# ============================================================
# Scheduler factory
# ============================================================


def build_scheduler(optimizer, args):
    """Build LR scheduler from parsed args, with optional warm-up."""
    if args.cos_restart_cyclic:
        base_sched = CosineAnnealingRestartCyclicLR(
            optimizer,
            periods=[args.nEpochs // 2, args.nEpochs // 2],
            restart_weights=[1.0, 0.5],
            eta_mins=[1e-6, 1e-7],
        )
    elif args.cos_restart:
        base_sched = CosineAnnealingRestartLR(
            optimizer,
            periods=[args.nEpochs],
            restart_weights=[1.0],
            eta_min=1e-6,
        )
    else:
        base_sched = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.scheduler_step, gamma=args.scheduler_gamma
        )

    if args.start_warmup and args.warmup_epochs > 0:
        return GradualWarmupScheduler(
            optimizer,
            multiplier=1.0,
            total_epoch=args.warmup_epochs,
            after_scheduler=base_sched,
        )
    return base_sched


# ============================================================
# Main epoch loop
# ============================================================


def main():
    # ------------------------------------------------------------------
    # Parse arguments
    # ------------------------------------------------------------------
    parser = option()
    args = parser.parse_args()

    from uwir.data.factory import (
        get_euvp_training_set,
        get_ufo120_training_set,
        get_uieb_training_set,
    )
    from uwir.losses import CompositeLoss
    from uwir.metrics import evaluate_loader

    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------
    device = torch.device("cuda" if (args.gpu_mode and torch.cuda.is_available()) else "cpu")
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # Reproducibility
    # ------------------------------------------------------------------
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if args.grad_detect:
        torch.autograd.set_detect_anomaly(True)

    # ------------------------------------------------------------------
    # Determine input channels and physics mode from model name
    # ------------------------------------------------------------------
    _, in_channels, physics_mode = parse_model_variant(args.model)
    physics_extractor = _resolve_physics_extractor(args.prior_method)
    physics_config = PhysicsConfig(
        guided_filter_radius=args.guided_filter_radius,
        guided_filter_eps=args.guided_filter_eps,
    )
    fusion_extractor = partial(compute_physics_features, config=physics_config)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = build_model(args.model, pretrained_backbone=args.pretrained_backbone).to(device)

    # ------------------------------------------------------------------
    # Multi-GPU  (DataParallel)
    # ------------------------------------------------------------------
    n_available = torch.cuda.device_count()
    n_gpus = min(args.num_gpus, n_available) if device.type == "cuda" else 1
    if n_gpus > 1:
        gpu_ids = list(range(n_gpus))
        model = nn.DataParallel(model, device_ids=gpu_ids)
        print(f"GPUs      : {n_gpus}  (DataParallel on {gpu_ids})")
    else:
        print("GPUs      : 1  (single)")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    print(f"Model     : {args.model}  (in_channels={in_channels}, physics={physics_mode})")
    print(f"Prior     : {args.prior_method}")

    # ------------------------------------------------------------------
    # Datasets & DataLoaders
    # ------------------------------------------------------------------
    def collate_fn_train(batch):
        return _collate_train(batch, physics_mode, physics_extractor, fusion_extractor)

    def collate_fn_val(batch):
        return _collate_val(batch, physics_mode, physics_extractor, fusion_extractor)

    # Training dataset
    if args.dataset == "euvp":
        train_ds = get_euvp_training_set(
            args.data_train_euvp,
            img_size=args.cropSize,
            subset=args.euvp_subset,
            in_memory=args.in_memory,
        )
    elif args.dataset == "uieb":
        train_ds = get_uieb_training_set(
            args.data_train_uieb, img_size=args.cropSize, in_memory=args.in_memory
        )
    elif args.dataset == "ufo120":
        train_ds = get_ufo120_training_set(
            args.data_train_euvp, img_size=args.cropSize, in_memory=args.in_memory
        )
    elif args.dataset == "euvp+uieb":
        euvp_ds = get_euvp_training_set(
            args.data_train_euvp,
            img_size=args.cropSize,
            subset=args.euvp_subset,
            in_memory=args.in_memory,
        )
        uieb_ds = get_uieb_training_set(
            args.data_train_uieb, img_size=args.cropSize, in_memory=args.in_memory
        )
        train_ds = data.ConcatDataset([euvp_ds, uieb_ds])
    else:
        raise ValueError(f"Unknown --dataset: {args.dataset}")

    # Validation: 10 % hold-out from the training set (paired → gives GT for
    # loss + metric computation).  Fixed seed for reproducibility.
    # NOTE: EUVP has no separate testA/testB; validation/ is unpaired (no GT).
    train_ds, val_ds = _split_train_validation(train_ds, args.split_seed)
    n_train, n_val = len(train_ds), len(val_ds)
    print(f"Dataset   : {args.dataset}  (train={n_train}, val={n_val})")
    loader_kwargs = _dataloader_kwargs(args, device)

    train_loader = data.DataLoader(
        train_ds,
        batch_size=args.batchSize,
        shuffle=args.shuffle,
        drop_last=True,
        collate_fn=collate_fn_train,
        **loader_kwargs,
    )

    val_loader = data.DataLoader(
        val_ds,
        batch_size=args.batchSize,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn_val,
        **loader_kwargs,
    )

    print(f"Train set : {len(train_ds)} samples  ({len(train_loader)} batches of {args.batchSize})")
    print(f"Val set   : {len(val_ds)} samples")

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------
    criterion = CompositeLoss(
        lambda_l1=args.L1_weight,
        lambda_perc=args.perceptual_weight,
        lambda_ssim=args.SSIM_weight,
        device=device,
    )

    # ------------------------------------------------------------------
    # Optimizer & Scheduler
    # ------------------------------------------------------------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = build_scheduler(optimizer, args)
    scaler = torch.amp.GradScaler(
        "cuda", enabled=args.amp and device.type == "cuda", init_scale=1024.0
    )

    print(f"Optimizer : AdamW  (lr={args.lr}, amp={scaler.is_enabled()})")
    print(
        f"Scheduler : {'CosineRestartCyclic' if args.cos_restart_cyclic else 'CosineRestart' if args.cos_restart else 'StepLR'}"
        f"  (warmup={args.warmup_epochs if args.start_warmup else 0} epochs)"
    )

    # ------------------------------------------------------------------
    # Checkpoint directory
    # If --run_name is set the directory is predictable (no timestamp),
    # which makes it easy to resume:  --resume <dir>/epoch_0040.pth
    # Without --run_name a timestamp is appended so parallel runs don't
    # overwrite each other.
    # ------------------------------------------------------------------
    start_epoch = 1
    RUN_TS = datetime.now().strftime("%Y%m%d_%H%M%S")
    RUN_ID = (
        f"{args.run_name.strip()}_{RUN_TS}"
        if args.run_name.strip()
        else f"{args.model}_{args.dataset}_{RUN_TS}"
    )
    CKPT_DIR = os.path.join(args.checkpoint_dir, RUN_ID)
    BEST_PATH = os.path.join(CKPT_DIR, "best_model.pth")
    LAST_PATH = os.path.join(CKPT_DIR, "last_model.pth")
    os.makedirs(CKPT_DIR, exist_ok=True)
    split_manifest = {
        "seed": args.split_seed,
        "train": _subset_manifest(train_ds),
        "validation": _subset_manifest(val_ds),
    }
    with open(os.path.join(CKPT_DIR, "split_manifest.json"), "w", encoding="utf-8") as file:
        json.dump(split_manifest, file, indent=2)

    # ---- log file (tee stdout → terminal + file) ---------------------------
    LOG_DIR = args.log_dir
    os.makedirs(LOG_DIR, exist_ok=True)
    LOG_PATH = os.path.join(LOG_DIR, f"{RUN_ID}.log")
    _log_file = open(LOG_PATH, "w", encoding="utf-8")
    _orig_stdout = sys.stdout
    sys.stdout = _Tee(_orig_stdout, _log_file)
    # ------------------------------------------------------------------------

    print(f"Run ID    : {RUN_ID}")
    print(f"Ckpt dir  : {CKPT_DIR}")
    print(f"Log file  : {LOG_PATH}")
    print(f"Command   : {' '.join(sys.argv)}")
    print(f"Timestamp : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 65)

    if args.resume and os.path.isfile(args.resume):
        start_epoch, _ = load_ckpt(args.resume, model, optimizer, device=str(device))
        start_epoch += 1
        print(f"Resumed   : {args.resume}  → continuing from epoch {start_epoch}")
    elif args.start_epoch > 0 and os.path.isfile(LAST_PATH):
        start_epoch, _ = load_ckpt(LAST_PATH, model, optimizer, device=str(device))
        start_epoch += 1
        print(f"Resumed   : {LAST_PATH}  → continuing from epoch {start_epoch}")

    # ------------------------------------------------------------------
    # Training state
    # ------------------------------------------------------------------
    history = {
        "_meta": {
            "model": args.model,
            "dataset": args.dataset,
            "euvp_subset": args.euvp_subset,
            "prior_method": args.prior_method,
            "seed": args.seed,
            "split_seed": args.split_seed,
            "train_samples": n_train,
            "validation_samples": n_val,
            "batch_size": args.batchSize,
            "crop_size": args.cropSize,
            "epochs": args.nEpochs,
            "learning_rate": args.lr,
            "num_gpus": n_gpus,
            "l1_weight": args.L1_weight,
            "perceptual_weight": args.perceptual_weight,
            "ssim_weight": args.SSIM_weight,
            "scheduler": (
                "cosine_restart_cyclic"
                if args.cos_restart_cyclic
                else "cosine_single_cycle"
                if args.cos_restart
                else "step"
            ),
        },
        "train_loss": [],
        "val_loss": [],
        "val_psnr": [],
        "val_ssim": [],
        "lr": [],
    }

    best_psnr = 0.0
    best_ssim = float("-inf")
    best_epoch = 0
    es = EarlyStopping(patience=args.early_stop_patience, min_delta=1e-4, mode="max")

    LOG_EVERY = max(1, args.nEpochs // 20)  # log ~20 times during training

    print(f"\n{'=' * 65}")
    print(f"  Training: {args.model}   epochs={args.nEpochs}   batch={args.batchSize}")
    print(f"{'=' * 65}")
    print(
        f"{'Epoch':>6}  {'TrainL':>8}  {'ValL':>8}  {'PSNR':>7}  {'SSIM':>7}  {'LR':>9}  {'Time':>6}"
    )
    print("-" * 65)

    train_start = time.time()

    for epoch in range(start_epoch, args.nEpochs + 1):
        t0 = time.time()

        # Train
        tr_loss, tr_comps = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            scaler=scaler,
            accumulation_steps=args.grad_accumulation_steps,
            log_batch_timing=args.log_batch_timing,
        )

        # Val loss (every epoch)
        vl_loss = val_loss_epoch(
            model, val_loader, criterion, device, amp_enabled=scaler.is_enabled()
        )

        # Validation metrics use the complete deterministic hold-out. Early
        # stopping advances only on epochs where these values are fresh.
        fresh_validation = epoch == 1 or epoch % args.val_interval == 0
        if fresh_validation:
            val_metrics, _ = evaluate_loader(model, val_loader, device)
            val_psnr = val_metrics["psnr"]
            val_ssim = val_metrics["ssim"]
        else:
            val_psnr = history["val_psnr"][-1] if history["val_psnr"] else 0.0
            val_ssim = history["val_ssim"][-1] if history["val_ssim"] else 0.0

        if not np.isfinite(val_psnr) or not np.isfinite(val_ssim):
            raise FloatingPointError(
                f"Non-finite validation metric at epoch {epoch}: PSNR={val_psnr}, SSIM={val_ssim}"
            )

        scheduler.step()
        cur_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        # Record
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["val_psnr"].append(val_psnr)
        history["val_ssim"].append(val_ssim)
        history["lr"].append(cur_lr)

        # Save best immediately when PSNR improves
        if (val_psnr, val_ssim) > (best_psnr, best_ssim):
            best_psnr = val_psnr
            best_ssim = val_ssim
            best_epoch = epoch
            save_ckpt(
                model,
                optimizer,
                epoch,
                {"psnr": val_psnr, "ssim": val_ssim, "val_loss": vl_loss},
                BEST_PATH,
            )
            flag = "  ← BEST ✓"
        else:
            flag = ""

        # Periodic checkpoint
        if epoch % args.snapshots == 0:
            save_ckpt(
                model,
                optimizer,
                epoch,
                {"psnr": val_psnr, "ssim": val_ssim},
                os.path.join(CKPT_DIR, f"epoch_{epoch:04d}.pth"),
            )

        # Log line
        if epoch == 1 or epoch % LOG_EVERY == 0 or flag:
            print(
                f"{epoch:>6}  {tr_loss:>8.4f}  {vl_loss:>8.4f}  "
                f"{val_psnr:>7.3f}  {val_ssim:>7.4f}  {cur_lr:>9.2e}  "
                f"{elapsed:>5.1f}s{flag}"
            )

        # Early stopping
        if fresh_validation and es(val_psnr):
            print(
                f"\nEarly stopping at epoch {epoch}  "
                f"(no improvement for {args.early_stop_patience} evals)"
            )
            break

    # Save last model
    save_ckpt(model, optimizer, epoch, {"psnr": val_psnr, "ssim": val_ssim}, LAST_PATH)

    total_min = (time.time() - train_start) / 60
    history["_meta"]["training_time_min"] = total_min
    history["_meta"]["peak_gpu_memory_mb"] = (
        torch.cuda.max_memory_allocated() / 1024**2 if device.type == "cuda" else None
    )
    print(f"\n{'=' * 65}")
    print(f"  Done.  Best PSNR: {best_psnr:.4f} dB at epoch {best_epoch}")
    print(f"  Total training time: {total_min:.1f} min")
    print(f"{'=' * 65}")

    # Export history JSON  –  saved alongside the checkpoints
    hist_path = os.path.join(CKPT_DIR, "training_history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    print("History saved →", hist_path)

    # ---- restore stdout and close log file ---------------------------------
    sys.stdout = _orig_stdout
    _log_file.close()
    print(f"[INFO] Training log saved → {LOG_PATH}")
    # ------------------------------------------------------------------------


if __name__ == "__main__":
    main()
