"""
setup_mamba.py
--------------
Automated helper to install causal-conv1d and mamba-ssm on Kaggle GPU environments
before running SGMA-Net training, enabling hardware-accelerated fused CUDA selective scan.
Supports pre-built binary wheels directly from GitHub releases for ultra-fast installation (seconds).
"""

import json
import os
import re
import subprocess
import sys
import urllib.request


# Known prebuilt wheels for quick resolution without GitHub API rate limits
KNOWN_WHEELS = {
    # Python 3.12, CUDA 12, ABI True
    ("cp312", "2.4", "cu12"): (
        "https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.5.4/causal_conv1d-1.5.4%2Bcu12torch2.4cxx11abiTRUE-cp312-cp312-linux_x86_64.whl",
        "https://github.com/state-spaces/mamba/releases/download/v2.3.0/mamba_ssm-2.3.0%2Bcu12torch2.4cxx11abiTRUE-cp312-cp312-linux_x86_64.whl",
    ),
    ("cp312", "2.5", "cu12"): (
        "https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.6.0/causal_conv1d-1.6.0%2Bcu12torch2.5cxx11abiTRUE-cp312-cp312-linux_x86_64.whl",
        "https://github.com/state-spaces/mamba/releases/download/v2.3.0/mamba_ssm-2.3.0%2Bcu12torch2.5cxx11abiTRUE-cp312-cp312-linux_x86_64.whl",
    ),
    ("cp312", "2.6", "cu12"): (
        "https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.7.0/causal_conv1d-1.7.0%2Bcu12torch2.6cxx11abiTRUE-cp312-cp312-linux_x86_64.whl",
        "https://github.com/state-spaces/mamba/releases/download/v2.3.2.post1/mamba_ssm-2.3.2.post1%2Bcu12torch2.6cxx11abiTRUE-cp312-cp312-linux_x86_64.whl",
    ),
    # Python 3.11, CUDA 12, ABI True
    ("cp311", "2.4", "cu12"): (
        "https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.5.4/causal_conv1d-1.5.4%2Bcu12torch2.4cxx11abiTRUE-cp311-cp311-linux_x86_64.whl",
        "https://github.com/state-spaces/mamba/releases/download/v2.3.0/mamba_ssm-2.3.0%2Bcu12torch2.4cxx11abiTRUE-cp311-cp311-linux_x86_64.whl",
    ),
    ("cp311", "2.5", "cu12"): (
        "https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.6.0/causal_conv1d-1.6.0%2Bcu12torch2.5cxx11abiTRUE-cp311-cp311-linux_x86_64.whl",
        "https://github.com/state-spaces/mamba/releases/download/v2.3.0/mamba_ssm-2.3.0%2Bcu12torch2.5cxx11abiTRUE-cp311-cp311-linux_x86_64.whl",
    ),
    ("cp311", "2.6", "cu12"): (
        "https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.7.0/causal_conv1d-1.7.0%2Bcu12torch2.6cxx11abiTRUE-cp311-cp311-linux_x86_64.whl",
        "https://github.com/state-spaces/mamba/releases/download/v2.3.2.post1/mamba_ssm-2.3.2.post1%2Bcu12torch2.6cxx11abiTRUE-cp311-cp311-linux_x86_64.whl",
    ),
    # Python 3.10, CUDA 12, ABI True
    ("cp310", "2.4", "cu12"): (
        "https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.5.4/causal_conv1d-1.5.4%2Bcu12torch2.4cxx11abiTRUE-cp310-cp310-linux_x86_64.whl",
        "https://github.com/state-spaces/mamba/releases/download/v2.3.0/mamba_ssm-2.3.0%2Bcu12torch2.4cxx11abiTRUE-cp310-cp310-linux_x86_64.whl",
    ),
    ("cp310", "2.5", "cu12"): (
        "https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.6.0/causal_conv1d-1.6.0%2Bcu12torch2.5cxx11abiTRUE-cp310-cp310-linux_x86_64.whl",
        "https://github.com/state-spaces/mamba/releases/download/v2.3.0/mamba_ssm-2.3.0%2Bcu12torch2.5cxx11abiTRUE-cp310-cp310-linux_x86_64.whl",
    ),
    ("cp310", "2.6", "cu12"): (
        "https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.7.0/causal_conv1d-1.7.0%2Bcu12torch2.6cxx11abiTRUE-cp310-cp310-linux_x86_64.whl",
        "https://github.com/state-spaces/mamba/releases/download/v2.3.2.post1/mamba_ssm-2.3.2.post1%2Bcu12torch2.6cxx11abiTRUE-cp310-cp310-linux_x86_64.whl",
    ),
}


def query_github_wheel(repo: str, py_tag: str, torch_ver: str, cu_tag: str, abi_tag: str = "TRUE") -> str | None:
    api_url = f"https://api.github.com/repos/{repo}/releases"
    req = urllib.request.Request(api_url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            releases = json.loads(resp.read().decode("utf-8"))
            for r in releases:
                for a in r.get("assets", []):
                    name = a.get("name", "")
                    if (
                        py_tag in name
                        and cu_tag in name
                        and f"torch{torch_ver}" in name
                        and "linux_x86_64" in name
                        and f"abi{abi_tag}" in name
                    ):
                        return a.get("browser_download_url")
    except Exception:
        pass
    return None


def get_env_info():
    py_ver = f"cp{sys.version_info.major}{sys.version_info.minor}"
    torch_ver = None
    cu_ver = "cu12"
    abi = "TRUE"

    try:
        import torch
        t_raw = torch.__version__
        m = re.match(r"^(\d+\.\d+)", t_raw)
        if m:
            torch_ver = m.group(1)
        if hasattr(torch, "_C") and hasattr(torch._C, "_GLIBCXX_USE_CXX11_ABI"):
            abi = "TRUE" if torch._C._GLIBCXX_USE_CXX11_ABI else "FALSE"
        if torch.cuda.is_available():
            cuda_v = torch.version.cuda or ""
            if cuda_v.startswith("11"):
                cu_ver = "cu11"
            elif cuda_v.startswith("12"):
                cu_ver = "cu12"
    except Exception:
        pass

    return py_ver, torch_ver, cu_ver, abi


def main():
    print("==================================================================", flush=True)
    print(">>> [UWIR Setup] Initializing hardware acceleration for SGMA-Net...", flush=True)
    print("==================================================================", flush=True)

    # Check if already installed
    try:
        from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
        print(">>> [UWIR Setup] mamba-ssm is already installed and loaded! Skipping installation.", flush=True)
        return
    except Exception:
        pass

    py_ver, torch_ver, cu_ver, abi = get_env_info()
    print(f">>> [UWIR Setup] Detected Environment: Python={py_ver}, Torch={torch_ver}, CUDA={cu_ver}, ABI={abi}", flush=True)

    installed = False

    # Attempt 1: Direct GitHub Release Prebuilt Wheels (Fastest & Most Reliable)
    causal_whl = None
    mamba_whl = None

    if torch_ver:
        key = (py_ver, torch_ver, cu_ver)
        if key in KNOWN_WHEELS and abi == "TRUE":
            causal_whl, mamba_whl = KNOWN_WHEELS[key]
            print(f">>> [UWIR Setup] Matched known prebuilt wheels for {key}:", flush=True)
        else:
            print(">>> [UWIR Setup] Querying GitHub releases for matching prebuilt wheels...", flush=True)
            causal_whl = query_github_wheel("Dao-AILab/causal-conv1d", py_ver, torch_ver, cu_ver, abi)
            mamba_whl = query_github_wheel("state-spaces/mamba", py_ver, torch_ver, cu_ver, abi)

    if causal_whl and mamba_whl:
        print(f"    Installing causal-conv1d from {causal_whl} ...", flush=True)
        print(f"    Installing mamba-ssm from {mamba_whl} ...", flush=True)
        cmd_wheel = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-build-isolation",
            causal_whl,
            mamba_whl,
        ]
        res = subprocess.run(cmd_wheel)
        if res.returncode == 0:
            installed = True

    # Attempt 2: Fallback to standard pip with --no-build-isolation
    if not installed:
        print(">>> [UWIR Setup] Prebuilt wheel install not completed. Trying PyPI with --no-build-isolation...", flush=True)
        cmd_pypi = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-build-isolation",
            "causal-conv1d>=1.4.0",
            "mamba-ssm>=2.2.0",
        ]
        res = subprocess.run(cmd_pypi)
        if res.returncode == 0:
            installed = True

    # Attempt 3: Final fallback standard pip
    if not installed:
        print(">>> [UWIR Setup] Retrying standard pip install...", flush=True)
        cmd_std = [sys.executable, "-m", "pip", "install", "causal-conv1d", "mamba-ssm"]
        subprocess.run(cmd_std)

    # Verification
    try:
        from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
        print("==================================================================", flush=True)
        print(">>> [UWIR Setup] SUCCESS: Official CUDA selective_scan_fn loaded successfully!", flush=True)
        print(">>> [UWIR Setup] SGMA-Net will run with fused CUDA acceleration (~1.5s/epoch).", flush=True)
        print("==================================================================", flush=True)
    except Exception as e:
        print("==================================================================", flush=True)
        print(f">>> [UWIR Setup] NOTICE: selective_scan_fn could not be imported ({e}).", flush=True)
        print(">>> [UWIR Setup] Training will proceed safely with native PyTorch scan fallback.", flush=True)
        print("==================================================================", flush=True)


if __name__ == "__main__":
    main()
