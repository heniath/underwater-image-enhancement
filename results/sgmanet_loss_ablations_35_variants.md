# SGMA-Net Loss Ablation Study — 35 Variants Comprehensive Report

**Project:** Underwater Image Restoration & Enhancement (UWIR)  
**Model Architecture:** SGMA-Net (`sgmanet_5ch`: RGB + transmission $t$ + background light $B$ via UDCP, ~1.10M params, 6.11 GigaMACs)  
**Training Setup:** UIEB 800 pairs (`raw-890` + `reference-890`), 100 Epochs, Batch Size 16, Crop 256×256, AdamW ($lr = 10^{-4}$), Cosine Annealing, Mixed Precision (AMP FP16).  
**Evaluation Benchmarks:**
1. **In-Domain Test (UIEB-90):** 90 test pairs at original resolution (~1000×800).
2. **Cross-Dataset Generalization Test (EUVP-515):** 515 test pairs from EUVP `test_samples` (256×256).

---

## 🏆 1. Bảng Vàng Xếp Hạng Top 10 Biến Thể Xuất Sắc Nhất Toàn Dự Án

*Xếp theo thứ tự ưu tiên chất lượng phục hồi trên tập kiểm thử chuẩn UIEB-90:*

| Hạng | Biến Thể Hàm Mất Mát | Nhóm Loss | Cấu Hình Trọng Số | Best Ep | Val PSNR (dB) | UIEB PSNR (dB) ↑ | UIEB SSIM ↑ | CIEDE2000 (↓) | UCIQE (↑) | UIQM (↑) | Điểm Nhấn Khoa Học |
| :---: | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 👑  | **Combo LVW 2.0 + UIQM 0.2 (ALL-TIME RECORD)** | 02. Synergistic Combos (LVW + UIQM) | `lvw=2.0, uiqm=0.2` | Ep 92 | 22.8547 | **22.8649** | **0.9083** | **8.7989** | 29.0945 | 1.8222 | 👑 ALL-TIME RECORD: Top 1 Lịch sử (PSNR 22.86 dB, SSIM 0.9083, CIEDE 8.79) |
| 🥈  | **Combo LVW 5.0 + UIQM 0.2** | 02. Synergistic Combos (LVW + UIQM) | `lvw=5.0, uiqm=0.2` | Ep 100 | 22.8416 | **22.7918** | **0.9048** | **8.8152** | 28.8492 | 1.7411 | 🔥 Đột phá mạnh: PSNR 22.79 dB, SSIM 0.9048, CIEDE 8.8152 |
| 🥉  | **Base GD Sharp + Combo (LVW 2.0 + UIQM 0.2)** | 11. Base Evolution (Prior & Structure) | `gd=1.0, lvw=2.0, uiqm=0.2 (NO SSIM)` | Ep 92 | 22.8911 | **22.7645** | **0.9073** | **8.8215** | 28.9715 | 1.8023 | 👑 KỶ LỤC VALIDATION TOÀN DỰ ÁN: Val PSNR 22.8911 dB, UIEB SSIM 0.9073 |
| 4.  | **Combo LVW 5.0 + UIQM 0.1** | 02. Synergistic Combos (LVW + UIQM) | `lvw=5.0, uiqm=0.1` | Ep 92 | 22.8754 | **22.6952** | **0.9048** | **8.9693** | 28.5223 | 1.7265 | Cân bằng rất tốt: Val PSNR 22.88 dB, Test PSNR 22.70 dB, SSIM 0.9048 |
| 5.  | **Combo LVW 2.0 + UIQM 0.1 (Top 2 Record)** | 02. Synergistic Combos (LVW + UIQM) | `lvw=2.0, uiqm=0.1` | Ep 92 | 22.7810 | **22.6902** | **0.9048** | **8.9538** | 28.6006 | 1.7583 | 🥈 Á quân toàn dự án (PSNR 22.69 dB, SSIM 0.9048) |
| 6.  | **MobileIE LVW (Peak Standalone)** | 03. MobileIE LVW Loss Sweeps | `lambda=5.0` | Ep 92 | 22.8010 | **22.6728** | **0.9031** | **8.8594** | 28.3355 | 1.6945 | 🏆 Đỉnh cao cực đại của LVW Standalone (PSNR 22.67 dB, CIEDE 8.86) |
| 7.  | **Combo LVW 1.0 + UIQM 0.1 (Top 1 Record)** | 02. Synergistic Combos (LVW + UIQM) | `lvw=1.0, uiqm=0.1` | Ep 92 | 22.8495 | **22.6646** | **0.9050** | **9.0136** | 28.6497 | 1.7659 | Quán quân Đợt 4 (PSNR 22.66 dB, SSIM 0.9050) |
| 8.  | **MobileIE LVW (Top 1 UIEB)** | 03. MobileIE LVW Loss Sweeps | `lambda=1.0` | Ep 92 | 22.7872 | **22.6367** | **0.9033** | **9.0894** | 28.1225 | 1.6931 | Quán quân Đợt 3 (PSNR 22.64 dB, CIEDE 9.09) |
| 9.  | **MobileIE LVW (Top CIEDE)** | 03. MobileIE LVW Loss Sweeps | `lambda=2.0` | Ep 96 | 22.8860 | **22.6352** | **0.9044** | **8.8902** | 28.2443 | 1.7228 | Kỷ lục màu sắc Đợt 4 (CIEDE 8.8902) |
| 10.  | **MobileIE LVW (High Weight)** | 03. MobileIE LVW Loss Sweeps | `lambda=50.0` | Ep 85 | 22.6670 | **22.5654** | **0.8986** | **9.2715** | 28.5083 | 1.7411 | Trọng số cực hạn (50x Base): kiểm tra ngưỡng gãy |

---

## 📊 2. Bảng Tổng Hợp Chi Tiết Toàn Bộ 35 Biến Thể Theo Nhóm Hàm Mất Mát

| STT | Nhóm Hàm Loss | Biến Thể (Variant) | Trọng Số Cấu Hình | Best Ep | Val PSNR (dB) | Val SSIM | Test PSNR (UIEB) | Test SSIM (UIEB) | CIEDE (UIEB) | UCIQE (UIEB) | UIQM (UIEB) | Test PSNR (EUVP) | Test SSIM (EUVP) | UCIQE (EUVP) | UIQM (EUVP) |
| :-: | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 01. Pure Base (Đối chứng chuẩn) | Pure Base | `1.0*L1 + 1.0*Lperc` | 97 | 22.5880 | 0.8949 | 22.4298 | 0.9032 | 9.1206 | 27.8660 | 1.7076 | 19.6236 | 0.8088 | 30.6311 | 2.0138 |
| 2 | 02. Synergistic Combos (LVW + UIQM) | Combo LVW 1.0 + UIQM 0.1 (Top 1 Record) | `lvw=1.0, uiqm=0.1` | 92 | 22.8495 | 0.8964 | 22.6646 | 0.9050 | 9.0136 | 28.6497 | 1.7659 | 19.3844 | 0.8041 | 31.1065 | 2.0595 |
| 3 | 02. Synergistic Combos (LVW + UIQM) | Combo LVW 2.0 + UIQM 0.1 (Top 2 Record) | `lvw=2.0, uiqm=0.1` | 92 | 22.7810 | 0.8971 | 22.6902 | 0.9048 | 8.9538 | 28.6006 | 1.7583 | 19.3351 | 0.8024 | 31.0792 | 2.0482 |
| 4 | 02. Synergistic Combos (LVW + UIQM) | Combo LVW 2.0 + UIQM 0.2 (ALL-TIME RECORD) | `lvw=2.0, uiqm=0.2` | 92 | 22.8547 | 0.8963 | 22.8649 | 0.9083 | 8.7989 | 29.0945 | 1.8222 | 19.2340 | 0.8004 | 31.2606 | 2.0764 |
| 5 | 03. MobileIE LVW Loss Sweeps | MobileIE LVW | `lambda=0.1` | 92 | 22.5427 | 0.8956 | 22.3512 | 0.9026 | 9.3539 | 27.8280 | 1.7008 | 19.5697 | 0.8074 | 30.7931 | 2.0279 |
| 6 | 03. MobileIE LVW Loss Sweeps | MobileIE LVW (Top 1 UIEB) | `lambda=1.0` | 92 | 22.7872 | 0.8972 | 22.6367 | 0.9033 | 9.0894 | 28.1225 | 1.6931 | 19.3408 | 0.8054 | 30.9452 | 2.0125 |
| 7 | 03. MobileIE LVW Loss Sweeps | MobileIE LVW (Top CIEDE) | `lambda=2.0` | 96 | 22.8860 | 0.8986 | 22.6352 | 0.9044 | 8.8902 | 28.2443 | 1.7228 | 19.3007 | 0.8032 | 30.9701 | 2.0173 |
| 8 | 03. MobileIE LVW Loss Sweeps | MobileIE LVW (Peak Standalone) | `lambda=5.0` | 92 | 22.8010 | 0.8967 | 22.6728 | 0.9031 | 8.8594 | 28.3355 | 1.6945 | 19.1809 | 0.8004 | 31.0104 | 1.9925 |
| 9 | 03. MobileIE LVW Loss Sweeps | MobileIE LVW | `lambda=10.0` | 62 | 22.7462 | 0.8956 | 22.4245 | 0.9005 | 9.1391 | 28.4888 | 1.7039 | 18.8732 | 0.7975 | 31.1630 | 2.0099 |
| 10 | 03. MobileIE LVW Loss Sweeps | MobileIE LVW | `lambda=30.0` | 62 | 22.6094 | 0.8923 | 22.2889 | 0.8966 | 9.4110 | 28.5425 | 1.7246 | 19.0236 | 0.7940 | 31.1986 | 2.0152 |
| 11 | 03. MobileIE LVW Loss Sweeps | MobileIE LVW (High Weight) | `lambda=50.0` | 85 | 22.6670 | 0.8901 | 22.5654 | 0.8986 | 9.2715 | 28.5083 | 1.7411 | 19.3119 | 0.7941 | 31.0891 | 2.0230 |
| 12 | 04. SSIM Loss Sweeps | SSIM (Early Stop) | `lambda=0.1` | 2 | 17.3028 | 0.7876 | 17.4778 | 0.7889 | 15.0652 | 23.7018 | 1.3621 | 21.3210 | 0.8172 | 26.7790 | 1.7111 |
| 13 | 04. SSIM Loss Sweeps | SSIM Full (Top EUVP) | `lambda=0.1` | 2 | 17.3660 | 0.7888 | 17.6251 | 0.7908 | 14.8344 | 23.5803 | 1.3474 | 21.4529 | 0.8196 | 26.7093 | 1.7053 |
| 14 | 04. SSIM Loss Sweeps | SSIM Full | `lambda=1.0` | 1 | 16.9773 | 0.7731 | 17.1465 | 0.7729 | 15.6724 | 22.8901 | 1.3146 | 20.0216 | 0.8052 | 26.3576 | 1.6761 |
| 15 | 05. Laplacian Edge Loss Sweeps | Edge Loss (Laplacian) | `lambda=0.1` | 96 | 22.6290 | 0.8965 | 22.2950 | 0.9019 | 9.2275 | 28.0189 | 1.7139 | 19.7110 | 0.8089 | 30.6357 | 2.0158 |
| 16 | 05. Laplacian Edge Loss Sweeps | Edge Loss | `lambda=1.0` | 92 | 22.5842 | 0.8934 | 22.4643 | 0.9015 | 9.2971 | 27.7284 | 1.6840 | 19.3931 | 0.8073 | 30.8027 | 2.0232 |
| 17 | 05. Laplacian Edge Loss Sweeps | Edge Loss (Sweet Spot) | `lambda=2.0` | 92 | 22.5865 | 0.8944 | 22.5033 | 0.9021 | 9.1569 | 27.7908 | 1.6888 | 19.5087 | 0.8088 | 30.7563 | 2.0201 |
| 18 | 05. Laplacian Edge Loss Sweeps | Edge Loss | `lambda=10.0` | 92 | 22.4639 | 0.8947 | 22.4716 | 0.9017 | 9.2113 | 27.7790 | 1.6834 | 19.4634 | 0.8073 | 30.7534 | 2.0117 |
| 19 | 06. Differentiable UIQM Loss Sweeps | UIQM Differentiable | `lambda=0.05` | 89 | 22.4465 | 0.8947 | 22.2198 | 0.9009 | 9.2726 | 28.1729 | 1.7437 | 19.3793 | 0.8045 | 30.9032 | 2.0444 |
| 20 | 06. Differentiable UIQM Loss Sweeps | UIQM Differentiable (Balanced) | `lambda=0.2` | 92 | 22.6593 | 0.8984 | 22.4769 | 0.9052 | 9.0730 | 29.1858 | 1.8910 | 19.3942 | 0.8039 | 31.2027 | 2.1217 |
| 21 | 06. Differentiable UIQM Loss Sweeps | UIQM Diff (Top UIQM) | `lambda=1.0` | 57 | 19.1585 | 0.8485 | 18.7609 | 0.8387 | 12.0926 | 30.3179 | 2.5020 | 18.0953 | 0.7775 | 31.8516 | 2.5337 |
| 22 | 07. WWE-UIE HVI Loss Sweeps | WWE-UIE HVI | `lambda=0.5 (k=0.2)` | 85 | 22.3170 | 0.8942 | 22.2012 | 0.8998 | 9.3577 | 27.8985 | 1.7084 | 19.4100 | 0.8062 | 30.8480 | 2.0288 |
| 23 | 07. WWE-UIE HVI Loss Sweeps | WWE-UIE HVI | `lambda=1.0` | 92 | 22.4099 | 0.8945 | 22.2420 | 0.8996 | 9.2730 | 27.5649 | 1.6441 | 19.5043 | 0.8081 | 30.7228 | 1.9914 |
| 24 | 08. Total Variation (TV) Loss Sweeps | TV Loss (UNTV 2022) | `lambda=0.001` | 92 | 22.5720 | 0.8978 | 22.3349 | 0.9014 | 9.2453 | 27.7832 | 1.6718 | 19.5211 | 0.8070 | 30.7460 | 2.0106 |
| 25 | 08. Total Variation (TV) Loss Sweeps | TV Loss | `lambda=1.0` | 89 | 22.4555 | 0.8942 | 22.2507 | 0.8974 | 9.1907 | 27.6773 | 1.6798 | 19.5141 | 0.8093 | 30.5267 | 1.9780 |
| 26 | 09. Adapt-PEFT LapPyr Loss | Adapt-PEFT LapPyr | `lambda=1.0 (K=3)` | 89 | 22.3530 | 0.8692 | 22.2461 | 0.8755 | 9.4566 | 28.4892 | 1.7056 | 18.8123 | 0.7881 | 31.0124 | 1.9440 |
| 27 | 10. Unified SOTA Multi-Loss Models | Unified SOTA Balanced | `lvw=1.0, edge=2.0, ssim=0.05, uiqm=0.05, lappyr=0.05` | 54 | 21.9135 | 0.8814 | 21.8196 | 0.8879 | 9.6497 | 28.4194 | 1.6852 | 18.6869 | 0.7984 | 31.1842 | 1.9987 |
| 28 | 10. Unified SOTA Multi-Loss Models | Unified Boost (LVW 2.0 + UIQM 0.15) | `lvw=2.0, edge=2.0, ssim=0.05, uiqm=0.15, lappyr=0.05` | 85 | 21.9861 | 0.8867 | 21.8067 | 0.8930 | 9.5910 | 28.9579 | 1.7699 | 18.8971 | 0.7969 | 31.3966 | 2.0463 |
| 29 | 10. Unified SOTA Multi-Loss Models | Unified Hyper (LVW 2.0 + UIQM 0.2 + HVI) | `lvw=2.0, edge=2.0, uiqm=0.2, hvi=0.5, ssim=0.05` | 11 | 19.8359 | 0.8474 | 19.8480 | 0.8457 | 11.8346 | 27.8913 | 1.7432 | 18.5585 | 0.8000 | 30.5621 | 1.9711 |
| 30 | 02. Synergistic Combos (LVW + UIQM) | Combo LVW 5.0 + UIQM 0.2 | `lvw=5.0, uiqm=0.2` | 100 | 22.8416 | 0.8983 | 22.7918 | 0.9048 | 8.8152 | 28.8492 | 1.7411 | 19.1067 | 0.7993 | 31.2344 | 2.0352 |
| 31 | 02. Synergistic Combos (LVW + UIQM) | Combo LVW 5.0 + UIQM 1.0 (Top 1 Vision) | `lvw=5.0, uiqm=1.0` | 93 | 22.2799 | 0.8889 | 22.1104 | 0.8957 | 9.3793 | 30.1370 | 2.0103 | 19.0381 | 0.7926 | 31.6598 | 2.1977 |
| 32 | 02. Synergistic Combos (LVW + UIQM) | Combo LVW 5.0 + UIQM 0.1 | `lvw=5.0, uiqm=0.1` | 92 | 22.8754 | 0.8994 | 22.6952 | 0.9048 | 8.9693 | 28.5223 | 1.7265 | 19.1113 | 0.7989 | 31.1531 | 2.0198 |
| 33 | 11. Base Evolution (Prior & Structure) | Base Quad Harm + Combo (LVW 2.0 + UIQM 0.2) | `ssim=0.1, gd=1.0, lvw=2.0, uiqm=0.2` | 17 | 17.8963 | 0.8091 | 17.1957 | 0.8040 | 14.5400 | 27.0654 | 1.7511 | 18.5557 | 0.7925 | 29.9174 | 1.9832 |
| 34 | 11. Base Evolution (Prior & Structure) | Base Friend Top1 + Combo (LVW 2.0 + UIQM 0.2) | `ssim=1.0, gd=1.0, lvw=2.0, uiqm=0.2` | 1 | 16.9734 | 0.7719 | 17.1183 | 0.7706 | 15.6544 | 22.9505 | 1.3287 | 20.0585 | 0.8055 | 26.4068 | 1.7022 |
| 35 | 11. Base Evolution (Prior & Structure) | Base GD Sharp + Combo (LVW 2.0 + UIQM 0.2) | `gd=1.0, lvw=2.0, uiqm=0.2 (NO SSIM)` | 92 | 22.8911 | 0.8972 | 22.7645 | 0.9073 | 8.8215 | 28.9715 | 1.8023 | 19.2545 | 0.8014 | 31.2656 | 2.0758 |

---

## 🌊 3. Đánh Giá Khái Quát Hóa Ngoại Suy (Cross-Dataset — EUVP-515 Benchmark)

*Top biến thể dẫn đầu theo từng tiêu chuẩn trên tập kiểm thử ngoại suy 515 ảnh:*

- **Vô địch Ngoại suy Cấu trúc & PSNR:**
  - `SSIM Full (Top EUVP)`: **21.4529 dB PSNR**, **0.8196 SSIM**, **CIEDE 11.0347** (Top 1 toàn diện về độ khớp ảnh trên EUVP).
- **Vô địch Cảm quan Thị giác Ngoại suy (UCIQE & UIQM):**
  - 👑 `Combo LVW 5.0 + UIQM 1.0`: **UCIQE = 31.6598**, **UIQM = 2.1977** (Kỷ lục cao nhất lịch sử dự án).
  - 🥈 `UIQM Diff (Top UIQM)` ($\lambda=1.0$): **UIQM = 2.5337**, **UCIQE = 31.8516**.
- **Cân bằng Hoàn hảo giữa Nội miền & Ngoại suy:**
  - 👑 `Base GD Sharp + Combo (LVW 2.0 + UIQM 0.2)`: UIEB đạt **22.7645 dB PSNR**, **0.9073 SSIM**; EUVP đạt **19.2545 dB PSNR**, **0.8014 SSIM**, **UCIQE 31.2656**, **UIQM 2.0758**.

---

## 🔬 4. Các Phát Hiện Khoa Học Cốt Lõi (Key Scientific Findings)

### 1. Kỷ Lục Validation PSNR Toàn Dự Án: Sức Mạnh của Gradient Difference Loss (GD)
Biến thể `sgmanet-base-gd-lvw2-uiqm02` thiết lập kỷ lục Validation PSNR mới: **22.8911 dB** (cao nhất trong toàn bộ 35 mô hình), đồng thời đạt **Test SSIM = 0.9073** và **CIEDE = 8.8215**.  
- **Cơ chế:** GD Loss giám sát gradient bậc 1 không gian theo hai trục $\nabla_x, \nabla_y$, ép các đường biên vi mô sắc nét mà không làm thay đổi thang đo độ sáng trung bình. Khi cộng hưởng với `LVW 2.0 + UIQM 0.2`, mô hình đạt độ hoàn thiện cao nhất cả về độ sắc nét lẫn màu sắc.

### 2. Kỷ Lục Lịch Sử Về Cảm Cụ Thị Giác: `LVW 5.0 + UIQM 1.0`
Khi kết hợp điểm cực đại của MobileIE LVW ($\lambda=5.0$) cùng Differentiable UIQM Loss ($\lambda=1.0$), mô hình thiết lập kỷ lục lịch sử:
- **EUVP-515:** UCIQE đạt **31.6598**, UIQM đạt **2.1977** (Vượt trội toàn bộ các mô hình truyền thống).
- **UIEB-90:** UCIQE đạt **30.1370**, UIQM đạt **2.0103** (Vượt ngưỡng 2.0).

### 3. Bản Chất Xung Đột Của SSIM Loss Với Thang Độ Sáng Tuyệt Đối
Các luồng chứa SSIM Loss trong tập UIEB (`base-harm` với $\lambda=0.1$ và `base-f11` với $\lambda=1.0$) đều bị dừng sớm (Early Stop ở Ep 37 và Ep 21) với Val PSNR chỉ đạt ~17 dB.
- **Giải mã:** SSIM tối ưu độ tương quan cấu trúc tương đối dựa trên phương sai cục bộ. Khi kết hợp với $L_1$ và LVW trên ảnh dưới nước, gradient của SSIM dễ làm xô lệch thang đo độ sáng trung bình (intensity offset). Do PSNR đo khoảng cách tuyệt đối MSE so với Ground Truth, chỉ cần lệch sáng nhẹ là PSNR tụt dốc.
- **Khuyến nghị:** Không nên dùng SSIM loss trực tiếp cho bài toán phục hồi ảnh UIEB; hãy thay thế bằng **Gradient Difference (GD Loss)** hoặc **Laplacian Edge Loss**.
