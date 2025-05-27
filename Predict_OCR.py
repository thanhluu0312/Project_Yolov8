import os
import cv2
from ultralytics import YOLO
import easyocr
import torch
import numpy as np

class IDCardProcessor:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"✅ Đang sử dụng thiết bị: {self.device.upper()}")

        # Thiết lập đường dẫn tuyệt đối tới mô hình
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, 'Project_Yolov8', 'runs_train', 'exp_selected15', 'weights', 'best.pt')

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ Không tìm thấy model tại: {model_path}")
        self.model = YOLO(model_path)

        # Khởi tạo OCR
        self.reader = easyocr.Reader(['vi', 'en'])

        # Nhãn lớp
        self.label_map = {
            0: 'avatar',
            1: 'name',
            2: 'id_number',
            3: 'dob',
            4: 'gender',
            5: 'nationality',
            6: 'address'
        }

    def preprocess_image(self, image):
        return cv2.resize(image, (640, 640))

    def enhance_text_region(self, crop):
        crop = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 11, 17, 17)
        return cv2.equalizeHist(gray)

    def process_image(self, image_path, save_debug=True):
        if not os.path.exists(image_path):
            return {"error": "Không tìm thấy ảnh"}

        # Đọc ảnh gốc
        img_original = cv2.imread(image_path)

        # Phát hiện bằng YOLO trên ảnh gốc
        results = self.model(img_original, conf=0.25)

        if len(results[0].boxes) == 0:
            return {"error": "Không phát hiện được thông tin"}

        boxes = results[0].boxes.xyxy.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()

        # Sắp xếp theo vị trí dọc
        sorted_items = sorted(zip(boxes, classes), key=lambda x: x[0][1])
        extracted_info = {}

        for box, cls_id in sorted_items:
            x1, y1, x2, y2 = map(int, box)
            crop = img_original[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            # Cải thiện vùng chữ và OCR
            enhanced_crop = self.enhance_text_region(crop)
            text = ' '.join(self.reader.readtext(enhanced_crop, detail=0)).strip()
            field = self.label_map.get(int(cls_id))

            if field != 'avatar' and text:
                extracted_info[field] = text

            # Vẽ box để debug
            if save_debug:
                cv2.rectangle(img_original, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img_original, f"{field}: {text}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if save_debug:
            debug_path = os.path.join(os.path.dirname(image_path), "output_detected.jpg")
            cv2.imwrite(debug_path, img_original)

        return extracted_info


if __name__ == "__main__":
    try:
        processor = IDCardProcessor()

        # Cập nhật đường dẫn chính xác đến ảnh test
        base_dir = os.path.dirname(os.path.abspath(__file__))
        test_img = os.path.join(base_dir, 'datasets', 'cccd', 'images', 'train', '001.jpg')

        print(f"\n🔍 Đang xử lý ảnh: {test_img}")
        results = processor.process_image(test_img)

        print("\n📋 Kết quả trích xuất:")
        if "error" in results:
            print(results["error"])
        else:
            for field, value in results.items():
                print(f"{field.upper()}: {value}")
    except Exception as e:
        print(f"❌ Lỗi: {str(e)}")
