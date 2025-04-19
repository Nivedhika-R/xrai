# https://docs.ultralytics.com/yolov5/tutorials/train_custom_data/#3-select-a-model 
from ultralytics import YOLO
import torch
import os 

# pretrained yolov8
model = YOLO("yolov8n.pt")

# train the model
yaml_data_path = "Server~/data.yaml"
model.train(data=yaml_data_path, epochs=50, imgsz=640, batch=16)

metrics = model.val(save_json=True)  # running validation
print(metrics.box.map)

#torch.save(model, "server/yolo-model.pt")