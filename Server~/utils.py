import cv2
import base64

def image2base64(image, png=False):
    _, buffer = cv2.imencode(".png", image) if png else cv2.imencode(".jpeg", image)
    img_bytes = buffer.tobytes()
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
    data_url = f"data:image/jpeg;base64,{img_base64}"
    return data_url
