import torch
from ultralytics import YOLO

class YoloHelper:
    def __init__(self, model_path):
        # self.model = YOLO("yolov8n.pt")
        self.model = YOLO(model_path)
        self.model.fuse()
        self.model.to('cuda' if torch.cuda.is_available() else 'cpu')

    def predict(self, image):
        results = self.model.predict(source=image)

        output = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0].item())
                class_name = self.model.names[cls_id]
                bbox = box.xyxy[0].tolist()
                output.append({
                    'class_name': class_name,
                    'bbox': bbox,
                    'confidence': float(box.conf[0].item())
                })

        return output
