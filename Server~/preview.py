import gradio as gr
import requests
import numpy as np
import cv2
import base64
from io import BytesIO
from PIL import Image
import time
import argparse


def fetch_latest_frame():
    try:
        response = requests.get(f"{SERVER_URL}/latest-frame", verify=VERIFY_SSL)
        data = response.json()
        img_base64 = data.get("image")
        if img_base64 is None:
            return None
        
        img_bytes = base64.b64decode(img_base64)
        img_np = np.frombuffer(img_bytes, np.uint8)
        img_cv2 = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
        
        return Image.fromarray(img_rgb)
    except Exception as e:
        print(f"Failed to fetch frame: {e}")
        return None

def fetch_latest_text():
    try:
        response = requests.get(f"{SERVER_URL}/llm-response", verify=VERIFY_SSL)
        data = response.json()
        text = data.get("llm_response")
        if text is None:
                return None
        return text
    except Exception as e:
        print(f"Failed to fetch text: {e}")
        return None

def live_stream():
    while True:
        frame = fetch_latest_frame()
        text = fetch_latest_text()
        if frame is not None and text is not None:
            yield frame, text
        elif frame is not None:
            yield frame, None
        elif text is not None:
            yield None, text
        else:
            yield None, None
        time.sleep(1)  # Poll every 100ms

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="XaiR Preview Window.")
    parser.add_argument('--ip', default='127.0.0.1', help='IP address to bind to')
    args = parser.parse_args()
    
    SERVER_URL = f"https://{args.ip}:8000"  # Or http://localhost:8000 if no SSL
    VERIFY_SSL = False  # Set to False if using self-signed certs
    
    with gr.Blocks() as demo:
                    image_display = gr.Image(type="pil")
                    text_display = gr.Textbox(label="LLM Response", lines=2, max_lines=5, interactive=False)
                    
                    demo.load(live_stream, [], [image_display, text_display])
    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=7861, share=False)
