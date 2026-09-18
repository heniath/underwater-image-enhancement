# Hướng Dẫn Cài Đặt & Chạy SGMA-Net Trên RTX 3060 (12GB VRAM)

Tài liệu này hướng dẫn chi tiết cách thiết lập môi trường và chạy thử nghiệm **SGMA-Net Loss Ablations** (Edge Loss trọng số 1, 2, 10 & Full SSIM Loss) trên máy tính cá nhân sử dụng GPU **NVIDIA GeForce RTX 3060**.

---

## 🚀 1. Tóm Tắt Nhanh (Quick Start)

Nếu máy đã có **Conda** / **Python 3.10-3.12** và **NVIDIA Driver**:

```bash
# 1. Clone hoặc Pull nhánh mới nhất
git checkout feat/sgmanet-loss-ablations
git pull origin feat/sgmanet-loss-ablations

# 2. Chạy script cài đặt tự động Mamba-SSM CUDA
bash scripts/setup_rtx3060.sh

# 3. Khởi chạy toàn bộ 4 luồng thử nghiệm
bash scripts/experiments/run_sgmanet_edge_and_ssim_runs.sh
```

---

## 📦 2. Cài Đặt Từng Bước (Chi Tiết)

### Bước 1: Tạo Môi Trường Conda (Khuyến nghị Python 3.11 hoặc 3.12)
```bash
conda create -n uwir python=3.11 -y
conda activate uwir
```

### Bước 2: Cài Đặt PyTorch Hỗ Trợ CUDA 12
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

### Bước 3: Cài Đặt Mamba-SSM & Causal-Conv1D (Siêu Tốc Qua Prebuilt Binary Wheels)
Trong repo đã tích hợp sẵn script phân tích môi trường và tải trực tiếp file wheel chính thức từ GitHub releases (chỉ mất ~5 giây):

```bash
python -m uwir.setup_mamba
```

*Hoặc cài trực tiếp bằng pip:*
```bash
pip install "causal-conv1d>=1.5.0"
pip install "mamba-ssm>=2.2.4"
```

### Bước 4: Cài Đặt Package UWIR & Thư Viện Đánh Giá
```bash
pip install -e .
pip install kornia thop tabulate pytest tqdm pillow
```

### Bước 5: Kiểm Tra Hoạt Động Của Mamba CUDA Kernel
```bash
python -c "
import torch
from uwir.models.sgmanet import build_sgmanet, MAMBA_CUDA_AVAILABLE

print('CUDA Device:', torch.cuda.get_device_name(0))
print('Mamba CUDA Fused Kernel Available:', MAMBA_CUDA_AVAILABLE)

model = build_sgmanet(in_channels=5).cuda().eval()
x = torch.randn(2, 5, 256, 256, device='cuda')
with torch.no_grad():
    out = model(x)
print('SGMA-Net output shape:', out.shape)
assert MAMBA_CUDA_AVAILABLE, 'Mamba CUDA kernel is required for fast training!'
print('>> ALL CHECKS PASSED!')
"
```

---

## 🧪 3. Khởi Chạy 4 Luồng Thử Nghiệm

Script tự động hoá [`scripts/experiments/run_sgmanet_edge_and_ssim_runs.sh`](scripts/experiments/run_sgmanet_edge_and_ssim_runs.sh) sẽ tuần tự chạy và ghi log:

1. **Luồng 1:** `Base + Edge Loss (weight = 1.0)`
2. **Luồng 2:** `Base + Edge Loss (weight = 2.0)`
3. **Luồng 3:** `Base + Edge Loss (weight = 10.0)`
4. **Luồng 4:** `Base + SSIM Loss (weight = 0.1, full 100 epochs, early_stop_patience = 100)`

Lệnh chạy:
```bash
bash scripts/experiments/run_sgmanet_edge_and_ssim_runs.sh
```

### Tuỳ biến tham số (nếu cần):
```bash
# Thay đổi đường dẫn dataset nếu khác mặc định:
DATA_UIEB="/path/to/UIEB" DATA_EUVP="/path/to/EUVP" bash scripts/experiments/run_sgmanet_edge_and_ssim_runs.sh
```

---

## ⚡ 4. Hiệu Năng Trên RTX 3060 (12GB VRAM)

- **VRAM Sử Dụng:** Chỉ ~1.8GB / 12GB (Rất mát và ổn định, có thể tăng batch size lên 32 nếu muốn).
- **Tốc Độ Huấn Luyện:** ~3 - 5 phút cho 100 epochs trên tập UIEB 800 ảnh (nhanh gấp đôi GPU T4 của Kaggle).
- **Đánh Giá Benchmark:** Tận dụng tối đa đa luồng CPU của máy local để tính các chỉ số CIEDE2000, UCIQE, UIQM nhanh hơn đáng kể so với môi trường Kaggle.
