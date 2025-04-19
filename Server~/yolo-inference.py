import torch
from ultralytics import YOLO
import numpy as np
from PIL import Image
import os

# trained, on weights - train 7/72
model_path = os.path("Server~/best.pt")
model = YOLO(model_path)

# loading, sample photo of multiple components
image_path = "insert image path from zip"
image = Image.open(image_path)

# inference
results = model(image)


for result in results:
    print(result.boxes)  # bounding boxes
    print(result.probs)  # probabilities of each class
    result.save("server/detected_output.jpg") # detected