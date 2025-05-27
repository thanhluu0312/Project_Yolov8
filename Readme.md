#  YOLOv8 - Trích xuất thông tin từ CCCD

Dự án sử dụng mô hình **YOLOv8** để phát hiện vùng thông tin trên thẻ **CCCD** (họ tên, số CMND/CCCD, ngày sinh, giới tính, quốc tịch, địa chỉ, ảnh chân dung).

---

##  Cấu trúc thư mục

```bash
Project_Yolov8/
├── datasets/
│   └── cccd/
│       ├── images/
│       │   ├── train/         # Ảnh dùng để huấn luyện
│       │   └── temp_train/    # Ảnh tạm tạo ra khi train
│       └── labels/
│           ├── train/         # Nhãn YOLO tương ứng với ảnh train
│           └── temp_train/    # Nhãn tạm
├── yolov8n.pt                 # Model YOLOv8 gốc
├── yolov8n.pt                 # Model đã fine-tuned
├── data.yaml                  # File cấu hình dataset cho YOLO
├── Train.py                   # Script huấn luyện mô hình YOLOv8
trích xuất thông tin
├── labelImg/                  # Công cụ tạo nhãn 
├── README.md
└── requirements.txt


```
## Cài Đặt
1. Clone kho lưu trữ:
```bash
git clone <https://github.com/thanhluu0312/Project_Yolov8.git>
cd Project_Yolov8
```
2. Cài đặt dependencies:
```bash
pip install ultralytics
```
3. Chạy script huấn luyện:
```bash
python Train.py
```


