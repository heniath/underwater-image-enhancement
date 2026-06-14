# ============================================================
# train.py  –  Training loop
# ============================================================

import json
import time

import torch
import torch.nn as nn

from config import cfg          # adjust import path as needed
from dataset import train_loader, val_loader
from model import model
from loss import criterion
from measure import evaluate_loader


# ============================================================
# Optimizer & Scheduler
# ============================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr           = cfg.LR,
    weight_decay = cfg.WEIGHT_DECAY,
)
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size = cfg.SCHEDULER_STEP,
    gamma     = cfg.SCHEDULER_GAMMA,
)
print(f"Optimizer : Adam  (lr={cfg.LR}, wd={cfg.WEIGHT_DECAY})")
print(f"Scheduler : StepLR  (step={cfg.SCHEDULER_STEP}, γ={cfg.SCHEDULER_GAMMA})")


# ============================================================
# EarlyStopping
# ============================================================

class EarlyStopping:
    def __init__(self, patience=15, min_delta=1e-4, mode="max"):
        self.patience   = patience
        self.min_delta  = min_delta
        self.mode       = mode
        self.counter    = 0
        self.best       = None
        self.stop       = False

    def __call__(self, score):
        if self.best is None:
            self.best = score
        elif (self.mode == "max" and score > self.best + self.min_delta) or \
             (self.mode == "min" and score < self.best - self.min_delta):
            self.best    = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
        return self.stop


# ============================================================
# Checkpoint helpers
# ============================================================

def save_ckpt(model, optimizer, epoch, metrics, path):
    torch.save({
        "epoch":     epoch,
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "metrics":   metrics,
    }, path)


def load_ckpt(path, model, optimizer=None, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    if optimizer and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt["epoch"], ckpt.get("metrics", {})


# ============================================================
# Inner-loop functions
# ============================================================

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    tot_loss = 0.0
    comps    = {"l1": 0.0, "perceptual": 0.0, "ssim_loss": 0.0}

    for inp, gt in loader:
        inp, gt = inp.to(device), gt.to(device)
        optimizer.zero_grad(set_to_none=True)
        pred         = model(inp)
        loss, parts  = criterion(pred, gt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        tot_loss += loss.item()
        for k in comps:
            comps[k] += parts.get(k, 0.0)

    n = len(loader)
    return tot_loss / n, {k: v / n for k, v in comps.items()}


@torch.no_grad()
def val_loss_epoch(model, loader, criterion, device):
    model.eval()
    tot = 0.0
    for inp, gt in loader:
        inp, gt = inp.to(device), gt.to(device)
        pred    = model(inp)
        loss, _ = criterion(pred, gt)
        tot    += loss.item()
    return tot / len(loader)


# ============================================================
# Main epoch loop  
# ============================================================

def main(device):
    history = {
        "train_loss": [], "val_loss": [],
        "val_psnr":   [], "val_ssim": [],
        "lr":         [],
    }

    best_psnr  = 0.0
    best_epoch = 0
    es         = EarlyStopping(patience=cfg.PATIENCE, min_delta=cfg.MIN_DELTA, mode="max")

    BEST_PATH = f"{cfg.OUTPUT_DIR}/checkpoints/best_model.pth"
    LAST_PATH = f"{cfg.OUTPUT_DIR}/checkpoints/last_model.pth"

    print(f"\n{'='*65}")
    print(f"  Training: {cfg.MODEL_NAME}   epochs={cfg.NUM_EPOCHS}   batch={cfg.BATCH_SIZE}")
    print(f"{'='*65}")
    print(f"{'Epoch':>6}  {'TrainL':>8}  {'ValL':>8}  {'PSNR':>7}  {'SSIM':>7}  {'LR':>9}  {'Time':>6}")
    print("-"*65)

    train_start = time.time()

    for epoch in range(1, cfg.NUM_EPOCHS + 1):
        t0 = time.time()

        # Train
        tr_loss, tr_comps = train_epoch(model, train_loader, optimizer, criterion, device)

        # Val loss (every epoch)
        vl_loss = val_loss_epoch(model, val_loader, criterion, device)

        # Val metrics (every LOG_EVERY epochs or epoch 1)
        if epoch == 1 or epoch % cfg.LOG_EVERY == 0:
            val_metrics, _ = evaluate_loader(model, val_loader, device, max_samples=100)
            val_psnr = val_metrics["psnr"]
            val_ssim = val_metrics["ssim"]
        else:
            val_psnr = history["val_psnr"][-1] if history["val_psnr"] else 0.0
            val_ssim = history["val_ssim"][-1] if history["val_ssim"] else 0.0

        scheduler.step()
        cur_lr  = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        # Record
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["val_psnr"].append(val_psnr)
        history["val_ssim"].append(val_ssim)
        history["lr"].append(cur_lr)

        # Save best immediately when PSNR improves
        if val_psnr > best_psnr + cfg.MIN_DELTA:
            best_psnr  = val_psnr
            best_epoch = epoch
            save_ckpt(model, optimizer, epoch,
                      {"psnr": val_psnr, "ssim": val_ssim, "val_loss": vl_loss},
                      BEST_PATH)
            flag = "  ← BEST ✓"
        else:
            flag = ""

        # Periodic checkpoint
        if epoch % cfg.SAVE_EVERY == 0:
            save_ckpt(model, optimizer, epoch,
                      {"psnr": val_psnr, "ssim": val_ssim},
                      f"{cfg.OUTPUT_DIR}/checkpoints/epoch_{epoch:04d}.pth")

        # Log line
        if epoch == 1 or epoch % cfg.LOG_EVERY == 0 or flag:
            print(f"{epoch:>6}  {tr_loss:>8.4f}  {vl_loss:>8.4f}  "
                  f"{val_psnr:>7.3f}  {val_ssim:>7.4f}  {cur_lr:>9.2e}  "
                  f"{elapsed:>5.1f}s{flag}")

        # Early stopping
        if es(val_psnr):
            print(f"\nEarly stopping at epoch {epoch}  (no improvement for {cfg.PATIENCE} evals)")
            break

    # Save last model
    save_ckpt(model, optimizer, epoch, {"psnr": val_psnr, "ssim": val_ssim}, LAST_PATH)

    total_min = (time.time() - train_start) / 60
    print(f"\n{'='*65}")
    print(f"  Done.  Best PSNR: {best_psnr:.4f} dB at epoch {best_epoch}")
    print(f"  Total training time: {total_min:.1f} min")
    print(f"{'='*65}")

    # Export history JSON
    with open(f"{cfg.OUTPUT_DIR}/training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print("History saved →", f"{cfg.OUTPUT_DIR}/training_history.json")


if __name__ == "__main__":
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main(device)