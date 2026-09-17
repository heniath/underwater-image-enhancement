"""
setup_mamba.py
--------------
Automated helper to install causal-conv1d and mamba-ssm on Kaggle GPU environments
before running SGMA-Net training, enabling hardware-accelerated fused CUDA selective scan.
"""

import subprocess
import sys


def main():
    print("==================================================================", flush=True)
    print(">>> [UWIR Setup] Installing causal-conv1d & mamba-ssm for SGMA-Net...", flush=True)
    print("==================================================================", flush=True)

    # Attempt 1: Install with --no-build-isolation
    cmd1 = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-build-isolation",
        "causal-conv1d>=1.4.0",
        "mamba-ssm>=2.2.0",
    ]
    ret = subprocess.run(cmd1)

    if ret.returncode != 0:
        print(f">>> [UWIR Setup] Attempt 1 exited with code {ret.returncode}. Retrying standard pip install...", flush=True)
        cmd2 = [sys.executable, "-m", "pip", "install", "causal-conv1d", "mamba-ssm"]
        ret = subprocess.run(cmd2)

    # Verification
    try:
        from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
        print(">>> [UWIR Setup] SUCCESS: Official CUDA selective_scan_fn loaded successfully! SGMA-Net will run with fused CUDA acceleration.", flush=True)
    except Exception as e:
        print(f">>> [UWIR Setup] NOTICE: selective_scan_fn could not be imported ({e}). Training will proceed with native PyTorch scan fallback.", flush=True)


if __name__ == "__main__":
    main()
