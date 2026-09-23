#!/usr/bin/env python3
import json
from pathlib import Path

TARGET_JOBS = [
    # Round 6
    ("sgmanet-combo-lvw5-uiqm02-uieb800", "Combo LVW 5.0 + UIQM 0.2", "thung192"),
    ("sgmanet-combo-lvw5-uiqm1-uieb800", "Combo LVW 5.0 + UIQM 1.0", "TThanh13"),
    ("sgmanet-combo-lvw5-uiqm01-uieb800", "Combo LVW 5.0 + UIQM 0.1", "RaH1111"),
    # Round 7
    ("sgmanet-base-harm-lvw2-uiqm02-uieb800", "Base Harm (SSIM 0.1+GD 1.0) + LVW 2.0 + UIQM 0.2", "thung192"),
    ("sgmanet-base-f11-lvw2-uiqm02-uieb800", "Base F11 (SSIM 1.0+GD 1.0) + LVW 2.0 + UIQM 0.2", "TThanh13"),
    ("sgmanet-base-gd-lvw2-uiqm02-uieb800", "Base GD (GD 1.0) + LVW 2.0 + UIQM 0.2", "RaH1111"),
]

outputs_dir = Path(__file__).resolve().parent / "outputs" / "sgmanet_loss_ablations"

print(f"{'Job Name':<38} | {'Best Ep':<7} | {'Val PSNR':<8} | {'Val SSIM':<8} | {'UIEB PSNR':<9} | {'UIEB SSIM':<9} | {'CIEDE':<7} | {'UCIQE':<7} | {'UIQM':<7} | {'EUVP PSNR':<9} | {'EUVP SSIM':<9} | {'Time(m)':<7}")
print("-" * 155)

for job_name, display_name, acc in TARGET_JOBS:
    job_dir = outputs_dir / job_name
    rf_list = list(job_dir.glob("**/test_results_all.json"))
    if not rf_list:
        print(f"{job_name:<38} | NOT DOWNLOADED / IN PROGRESS")
        continue
    rf = rf_list[0]
    try:
        with open(rf, "r") as f:
            data = json.load(f)
        runs = data.get("runs", {})
        for run_id, rdata in runs.items():
            best_ep = rdata.get("best_epoch", "-")
            ckpt = rdata.get("ckpt_metrics", {})
            val_psnr = ckpt.get("psnr", 0.0)
            val_ssim = ckpt.get("ssim", 0.0)
            tr_time = rdata.get("training_time_min", 0.0)
            bench = rdata.get("benchmarks", {})
            uieb = bench.get("uieb", {})
            euvp = bench.get("euvp", {})
            print(f"{job_name:<38} | {best_ep:<7} | {val_psnr:<8.4f} | {val_ssim:<8.4f} | {uieb.get('psnr', 0.0):<9.4f} | {uieb.get('ssim', 0.0):<9.4f} | {uieb.get('ciede2000', 0.0):<7.4f} | {uieb.get('uciqe', 0.0):<7.4f} | {uieb.get('uiqm', 0.0):<7.4f} | {euvp.get('psnr', 0.0):<9.4f} | {euvp.get('ssim', 0.0):<9.4f} | {tr_time:<7.1f}")
    except Exception as e:
        print(f"{job_name:<38} | Error: {e}")
