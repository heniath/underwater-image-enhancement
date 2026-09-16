#!/usr/bin/env bash
# ==============================================================================
# setup_wsl_mamba.sh
# ------------------
# Automated setup for Mamba-SSM (CUDA) on WSL2 (Ubuntu) for SGMA-Net / UWIR
# ==============================================================================

set -e

echo "=== [1/6] Checking NVIDIA GPU Passthrough in WSL2 ==="
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
    echo ">> NVIDIA GPU detected successfully in WSL2!"
else
    echo ">> WARNING: nvidia-smi not found. Ensure NVIDIA Windows GPU drivers are up to date."
fi

echo ""
echo "=== [2/6] Installing Essential System Dependencies ==="
sudo apt-get update -y
sudo apt-get install -y build-essential curl wget git

echo ""
echo "=== [3/6] Setting up Miniconda (if not already installed) ==="
CONDA_DIR="$HOME/miniconda3"
if [ ! -d "$CONDA_DIR" ]; then
    echo ">> Downloading Miniconda installer..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p "$CONDA_DIR"
    rm miniconda.sh
    echo ">> Miniconda installed to $CONDA_DIR"
fi

# Source conda environment
source "$CONDA_DIR/etc/profile.d/conda.sh"
conda init bash > /dev/null 2>&1 || true

echo ""
echo "=== [4/6] Creating / Activating Conda Environment 'uwir-mamba' ==="
if conda info --envs | grep -q "uwir-mamba"; then
    echo ">> Environment 'uwir-mamba' already exists. Activating..."
else
    echo ">> Creating conda environment 'uwir-mamba' with Python 3.11..."
    conda create -n uwir-mamba python=3.11 -y
fi
conda activate uwir-mamba

echo ""
echo "=== [5/6] Installing PyTorch with CUDA, causal-conv1d, and mamba-ssm ==="
pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Install causal-conv1d and mamba-ssm prebuilt wheels for Linux
pip install "causal-conv1d>=1.4.0"
pip install "mamba-ssm>=2.2.0"

echo ""
echo "=== [6/6] Installing UWIR project dependencies ==="
PROJECT_DIR="/mnt/d/eureka/underwater-image-enhancement"
if [ -d "$PROJECT_DIR" ]; then
    cd "$PROJECT_DIR"
    pip install -e .
    pip install pytest tabulate thop
    
    echo ""
    echo "=== Verifying Mamba CUDA & SGMA-Net in WSL2 ==="
    python -c "
import torch
print('PyTorch Version:', torch.__version__)
print('CUDA Available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU Device:', torch.cuda.get_device_name(0))

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    print('>> SUCCESS: Official mamba_ssm CUDA selective_scan_fn is loaded and ready!')
except Exception as e:
    print('>> ERROR loading mamba_ssm:', e)

from uwir.models.sgmanet import build_sgmanet, MAMBA_CUDA_AVAILABLE
print('MAMBA_CUDA_AVAILABLE in SGMA-Net:', MAMBA_CUDA_AVAILABLE)

model = build_sgmanet(in_channels=3).cuda()
x = torch.randn(2, 3, 256, 256, device='cuda')
out = model(x)
print('SGMA-Net forward pass success! Output shape:', out.shape)
"
    pytest tests/test_sgmanet.py -v
    echo ""
    echo "=========================================================="
    echo " [DONE] Mamba CUDA setup completed successfully in WSL2!  "
    echo "=========================================================="
else
    echo ">> Project directory $PROJECT_DIR not found. Mount might differ."
fi
