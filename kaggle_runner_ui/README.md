# ⚡ Kaggle Multi-Account & Multi-Thread OCR Control Hub

High-Performance Modern Web Application for managing multiple Kaggle accounts, tracking weekly GPU quota (30h/week), controlling parallel batch jobs, analyzing benchmark metrics, and interacting with the built-in **Antigravity AI Assistant**.

---

## 🚀 Quick Start

### 1. Cài đặt Phụ thuộc
```bash
cd kaggle_runner_ui
python -m pip install -r requirements.txt
```

### 2. Khởi chạy Web Server
```bash
python server.py --port 8080
```
Hoặc trên Windows, bạn chỉ cần **click đúp chuột** vào file:
```text
run_windows.bat
```

Trình duyệt sẽ tự động mở tại địa chỉ: `http://127.0.0.1:8080`

---

## 🌟 Tính năng Nổi bật

1. **⚡ Dashboard & Quotas:**
   - Theo dõi GPU Quota tuần (30h/tài khoản) và Active Batch GPU Slots (tối đa 2 slots/tài khoản).
   - Kiểm tra danh sách kernel đang chạy trực tiếp trên Kaggle.
   - Thêm / Xóa / Cập nhật Token tài khoản tức thì.
2. **🧵 Multi-Thread Manager:**
   - Quản lý ma trận các luồng (jobs) độc lập trên từng tài khoản.
   - Kích hoạt chạy toàn bộ luồng song song (Multi-Account Parallel Dispatcher).
   - Thêm luồng mới với tùy chọn Benchmark hoặc Finetuning (10 Epochs).
3. **🤖 Antigravity AI Companion (Chatbot):**
   - Điều khiển hệ thống bằng ngôn ngữ tự nhiên (tiếng Việt / tiếng Anh).
   - Tự động phân tích lỗi trong file log Kaggle và đưa ra giải pháp.
4. **📊 Benchmark Analytics:**
   - Biểu đồ tương tác Chart.js cho CER (Character Error Rate), WER (Word Error Rate), và Tốc độ (Ký tự / Giây).
   - Bảng đối soát chi tiết giữa mô hình Pretrained và Finetuned.
5. **📜 Live Logs & Debug:**
   - Đọc log thời gian thực, lọc từ khóa, chẩn đoán nguyên nhân crash.
6. **⚙️ Git Sync:**
   - Kiểm tra `git status` và thực hiện `git push` trực tiếp từ giao diện web.
