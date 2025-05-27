import os
import shutil
from ultralytics import YOLO

# Lấy đường dẫn tuyệt đối của thư mục Project_Yolov8
current_dir = os.path.dirname(os.path.abspath(__file__))  # datasets/Train.py
project_root = os.path.abspath(os.path.join(current_dir, '..'))  # Project_Yolov8/

# Đường dẫn tới ảnh và label gốc
img_dir = os.path.join(project_root, 'datasets', 'cccd', 'images', 'train')
label_dir = os.path.join(project_root, 'datasets', 'cccd', 'labels', 'train')
# Folder tạm
temp_img_dir = os.path.join(project_root, 'datasets', 'cccd', 'images', 'temp_train')
temp_label_dir = os.path.join(project_root, 'datasets', 'cccd', 'labels', 'temp_train')
# Tạo lại thư mục tạm
if os.path.exists(temp_img_dir):
    shutil.rmtree(temp_img_dir)
os.makedirs(temp_img_dir)

if os.path.exists(temp_label_dir):
    shutil.rmtree(temp_label_dir)
os.makedirs(temp_label_dir)

# Liệt kê các file ảnh
images = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
print("Danh sách ảnh:")
for idx, img in enumerate(images):
    print(f"{idx}: {img}")

# Yêu cầu người dùng chọn ảnh
selected_indices = input("Nhập hình ảnh muốn train: ")
selected_indices = [int(i.strip()) for i in selected_indices.split(',')]

# Copy ảnh + label tương ứng sang folder tạm
for idx in selected_indices:
    img_name = images[idx]
    base_name = os.path.splitext(img_name)[0]
    label_name = f"{base_name}.txt"

    shutil.copy(os.path.join(img_dir, img_name), temp_img_dir)
    shutil.copy(os.path.join(label_dir, label_name), temp_label_dir)

# Train với dữ liệu tạm
model = YOLO(os.path.join(project_root, 'yolov8n.pt'))
model.train(
    data=os.path.join(project_root, 'datasets', 'cccd', 'data_temp.yaml'),
    epochs=50,
    imgsz=1280,
    batch=16,
    project=os.path.join(project_root, 'Project_Yolov8', 'runs_train'),
    name='exp_selected'
)
