import time
import io
import cv2
import base64
import numpy as np

from PIL import Image

class Frame:
    def __init__(self, img, ID):
        self.img = img
        self.frameID = ID
        self.timestamp = time.time()


def image2base64(image, png=False):
    _, buffer = cv2.imencode(".png", image) if png else cv2.imencode(".jpeg", image)
    img_bytes = buffer.tobytes()
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
    data_url = f"data:image/jpeg;base64,{img_base64}"
    return data_url

def image2base64_ollama(image, png=False):
    _, buffer = cv2.imencode(".png", image) if png else cv2.imencode(".jpeg", image)
    img_bytes = buffer.tobytes()
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
    #data_url = f"data:image/jpeg;base64,{img_base64}"
    return img_base64
