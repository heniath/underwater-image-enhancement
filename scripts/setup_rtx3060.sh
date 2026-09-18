#!/usr/bin/env bash
# ==============================================================================
# setup_rtx3060.sh
# ----------------
# One-click automated setup for SGMA-Net + Mamba-SSM (CUDA Fused Kernel)
# on Local Machine with NVIDIA GeForce RTX 3060 (12GB VRAM)
# Works for Linux / WSL2 (Ubuntu).
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

echo "=================================================================="
echo ">>> [1/5] Checking NVIDIA RTX 3060 GPU and CUDA Support"
echo "=================================================================="
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
    echo ">> NVIDIA GPU detected successfully!"
else
    echo ">> [ERROR] nvidia-smi not found! Please install/update NVIDIA Drivers."
    exit 1
fi

echo ""
echo "=================================================================="
echo ">>> [2/5] Ensuring PyTorch with CUDA 12.x is Installed"
echo "=================================================================="
python3 -c "
import torch
print('Python version  :', torch.sys.version.split()[0])
print('PyTorch version :', torch.__version__)
print('CUDA Available  :', torch.cuda.is_available())
if torch.cuda.is_available():
    print('Device Name     :', torch.cuda.get_device_name(0))
" || {
    echo ">> PyTorch with CUDA not found. Installing PyTorch 2.4+ cu124..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
}

echo ""
echo "=================================================================="
echo ">>> [3/5] Installing causal-conv1d & mamba-ssm (Official Binary Wheels)"
echo "=================================================================="
python3 -m uwir.setup_mamba

echo ""
echo "=================================================================="
echo ">>> [4/5] Installing UWIR Package & Evaluation Dependencies"
echo "=================================================================="
pip install --upgrade pip
pip install -e .
pip install kornia thop tabulate pytest tqdm pillow torchvision

echo ""
echo "=================================================================="
echo ">>> [5/5] Verifying SGMA-Net CUDA Fused Kernel on RTX 3060"
echo "=================================================================="
python3 -c "
import torch
from uwir.models.sgmanet import build_sgmanet, MAMBA_CUDA_AVAILABLE

print('MAMBA_CUDA_AVAILABLE in SGMA-Net:', MAMBA_CUDA_AVAILABLE)
assert torch.cuda.is_available(), 'CUDA is not available!'

model = build_sgmanet(in_channels=5).cuda().eval()
x = torch.randn(2, 5, 256, 256, device='cuda')
with torch.no_grad():
    out = model(x)
print('>> SGMA-Net forward pass SUCCESS! Output shape:', out.shape)
"

pytest tests/test_sgmanet.py -v

echo ""
echo "=================================================================="
echo " [SUCCESS] RTX 3060 Environment is 100% Ready!"
echo " To run Edge Loss (1, 2, 10) & Full SSIM experiments:"
echo "   bash scripts/experiments/run_sgmanet_edge_and_ssim_runs.sh"
echo "=================================================================="
