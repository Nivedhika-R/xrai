import requests
import numpy as np
import cv2
import base64
import time
import argparse

from PIL import Image
import gradio as gr


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

def fetch_llm_images(): # i.e. the 2 images that generated the text (croppped user image + sample image)
    try:
        response = requests.get(f"{SERVER_URL}/llm-images", verify=VERIFY_SSL)
        data = response.json()

        img_base64 = data.get("user_image")
        if img_base64 is None:
            return None, None, None
        img_bytes = base64.b64decode(img_base64)
        img_np = np.frombuffer(img_bytes, np.uint8)
        img_cv2 = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
        user_image = Image.fromarray(img_rgb)

        img_base64 = data.get("yolo_image")
        if img_base64 is None:
            return None, None, None
        img_bytes = base64.b64decode(img_base64)
        img_np = np.frombuffer(img_bytes, np.uint8)
        img_cv2 = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
        yolo_image = Image.fromarray(img_rgb)

        img_base64 = data.get("sample_image")
        if img_base64 is None:
            return None, None, None
        img_bytes = base64.b64decode(img_base64)
        img_np = np.frombuffer(img_bytes, np.uint8)
        img_cv2 = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        # img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
        sample_image = Image.fromarray(img_cv2)

        return user_image, yolo_image, sample_image
    except Exception as e:
        print(f"Failed to fetch frame: {e}")
        return None, None, None

def live_stream():
    while True:
        frame = fetch_latest_frame()
        text = fetch_latest_text()
        user_image, yolo_image, sample_image = fetch_llm_images()
        yield frame, text, user_image, yolo_image, sample_image
        # time.sleep(0.05)  # Poll every 100ms

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="XaiR Preview Window.")
    parser.add_argument('--ip', default='127.0.0.1', help='IP address to bind to')
    args = parser.parse_args()

    SERVER_URL = f"https://{args.ip}:8000"  # Or http://localhost:8000 if no SSL
    VERIFY_SSL = False  # Set to False if using self-signed certs

    with gr.Blocks(css=".big-textbox textarea {font-size: 18px !important;}") as demo:
        # add text title
        gr.Markdown("<h1 style='text-align: center;'>XaiR Preview Window</h1>")

        with gr.Row():
            with gr.Column(scale = 1):
                gr.Markdown("<h2 style='text-align: left;'>Live Stream w/ Obj Detection:</h2>")
                image_display = gr.Image(type="pil", show_label=False)

                gr.Markdown("<h2 style='text-align: left;'>Annotated Image w/ Obj Detection [Sent to LLM]:</h2>")
                yolo_image_display = gr.Image(type="pil", show_label=False)
            with gr.Column(scale = 1):

                gr.Markdown("<h2 style='text-align: left;'>Current Image [Sent to LLM]:</h2>")
                user_image_display = gr.Image(type="pil", show_label=False)

                gr.Markdown("<h2 style='text-align: left;'>Reference Image of Completed Step [Sent to LLM]:</h2>")
                sample_image_display = gr.Image(type="pil", show_label=False)

            with gr.Column(scale = 1):
                gr.Markdown("<h2 style='text-align: left;'>Response from LLM:</h2>")
                #increase font size of textbox
                text_display = gr.Textbox( lines=5, max_lines=8, interactive=False, show_label = False, elem_classes="big-textbox")

        demo.load(live_stream, [], [image_display, text_display, user_image_display, yolo_image_display, sample_image_display])
    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=7861, share=False)
