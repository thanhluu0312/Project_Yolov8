#  YOLOv8 - Object Detection on Vietnamese ID Cards (CCCD)
The project uses the **YOLOv8** model to detect information areas on the **CCCD** card (full name, ID card/CCCD number, date of birth, gender, nationality, address, portrait photo).

---

##  Directory Structure

```bash
Project_Yolov8/
├── datasets
│   └── cccd
│       ├── images
│       │   ├── train
│       │   │   ├── img1.jpg
│       │   │   ├── img2.jpg
│       │   │   └── ...
│       ├── labels
│       │   ├── train
│       │   │   ├── img1.txt
│       │   │   ├── img2.txt
│       │   │   └── ...
│       └── data_temp.yaml
├── Train.py
└── ...

```
## Workflow
 
The project workflow is straightforward: Given an input image, the YOLOv8 model detects regions of interest. Users can select specific images for training, and the model is fine-tuned on the chosen data. Training results are saved in the Project_Yolov8/runs_train/ directory.

```bash
Original Image --> [Inference] --> YOLOv8 Object Detector --> [Train] --> Saved Model
```
## Installation
1. Clone the repository::
```bash
git clone <https://github.com/thanhluu0312/Project_Yolov8.git>
cd Project_Yolov8
```
2. Install dependencies:
```bash
pip install ultralytics
```
3. Run the training script:
```bash
python Train.py
```
## Results
1. Training Details:
```
Model: YOLOv8 Small
Fine-tuned on a custom CCCD dataset
Training Epochs: 50
Losses:
-Train Box Loss: 1.305
-Validation Box Loss: 1.2908
-Classification Loss (cls_loss): Smooth L1 loss
-Total Loss: Gradual decrease, stabilizing around 0.95
```
2. Performance Metrics:
```
Mean Average Precision (mAP@0.5): 0.995
mAP@0.5:0.95: 0.531
```
3. Precision and Recall:
```
Precision: 0.995
Recall: 0.997
