from io import BytesIO
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
        print("HERE")

        return Image.fromarray(img_rgb)
    except Exception as e:
        print(f"Failed to fetch frame: {e}")
        return None

# def fetch_latest_text():
#     try:
#         response = requests.get(f"{SERVER_URL}/llm-response", verify=VERIFY_SSL)
#         data = response.json()
#         text = data.get("llm_response")
#         if text is None:
#                 return None
#         return text
#     except Exception as e:
#         print(f"Failed to fetch text: {e}")
#         return None

# def fetch_llm_images(): # i.e. the 2 images that generated the text (croppped user image + sample image)
#     try:
#         response = requests.get(f"{SERVER_URL}/llm-images", verify=VERIFY_SSL)
#         data = response.json()

#         img_base64 = data.get("user_image")
#         if img_base64 is None:
#             return None, None, None
#         img_bytes = base64.b64decode(img_base64)
#         img_np = np.frombuffer(img_bytes, np.uint8)
#         img_cv2 = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
#         img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
#         user_image = Image.fromarray(img_rgb)

#         img_base64 = data.get("yolo_image")
#         if img_base64 is None:
#             return None, None, None
#         img_bytes = base64.b64decode(img_base64)
#         img_np = np.frombuffer(img_bytes, np.uint8)
#         img_cv2 = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
#         img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
#         yolo_image = Image.fromarray(img_rgb)

#         img_base64 = data.get("sample_image")
#         if img_base64 is None:
#             return None, None, None
#         img_bytes = base64.b64decode(img_base64)
#         img_np = np.frombuffer(img_bytes, np.uint8)
#         img_cv2 = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
#         # img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
#         sample_image = Image.fromarray(img_cv2)

#         return user_image, yolo_image, sample_image
#     except Exception as e:
#         print(f"Failed to fetch frame: {e}")
#         return None, None, None

def live_stream():
     while True:
         frame = fetch_latest_frame()
         #text = fetch_latest_text()
         #user_image, yolo_image, sample_image = fetch_llm_images()
<<<<<<< HEAD
         yield frame
         time.sleep(0.05)  # Poll every 100ms
=======
         time.sleep(0.05)  # Poll every 100ms
         yield frame,frame
>>>>>>> d2b7fe9febc15c6eb900f232aa75779667e41bd8

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="XaiR Preview Window.")
    parser.add_argument('--ip', default='127.0.0.1', help='IP address to bind to')
    args = parser.parse_args()

    SERVER_URL = f"https://{args.ip}:7860"  # Or http://localhost:8000 if no SSL
    VERIFY_SSL = False  # Set to False if using self-signed certs

    with gr.Blocks(css=".big-textbox textarea {font-size: 18px !important;}") as demo:

        demo.queue()

        test_img = gr.Image(type="pil", label="Fetch Test Result")
        test_btn = gr.Button("Fetch Latest Frame")
        test_btn.click(fn=fetch_latest_frame, inputs=None, outputs=test_img)
        # add text title
        gr.Markdown("<h1 style='text-align: center;'>XaiR Preview Window</h1>")

<<<<<<< HEAD
        #gr.Markdown("<h2 style='text-align: left;'>Live Stream w/ Obj Detection:</h2>")
        #image_display = gr.Image(type="pil", show_label=False)
        with gr.Row():
            # Left: live stream
=======
        #cOMMRNTED THE 2 LINES BELOW FOR ANNOTATION
        #gr.Markdown("<h2 style='text-align: left;'>Live Stream w/ Obj Detection:</h2>")
        # image_display = gr.Image(type="pil", show_label=False)
        with gr.Row():
>>>>>>> d2b7fe9febc15c6eb900f232aa75779667e41bd8
            with gr.Column():
                gr.Markdown("### Live Stream Feed")
                live_display = gr.Image(type="pil", interactive=False, show_label=False)

<<<<<<< HEAD
            # Right: annotator
            with gr.Column():
                gr.Markdown("### Draw / Click to Annotate")
                annotator = gr.ImageEditor(
                    type="pil",
                    brush=gr.Brush(),       # enables freehand + point
                    interactive=True,
                    show_label=False
                )
        demo.load(live_stream, [], outputs=[live_display, annotator])

        # optional: a button to send the annotation back to your server
        def send_annotation(annotated_img):
            # e.g. convert to base64 and POST to your endpoint
            buffered = BytesIO()
            annotated_img.save(buffered, format="PNG")
            b64 = base64.b64encode(buffered.getvalue()).decode()
            requests.post(f"{SERVER_URL}/submit-annotation",
                          json={"image": b64},
                          verify=VERIFY_SSL)
            return "Sent!"

        send_btn = gr.Button("Save & Send Annotation")
        result_txt = gr.Textbox(interactive=False)
        send_btn.click(
            fn=send_annotation,
            inputs=[annotator],
            outputs=[result_txt]
        )

=======
                capture_btn = gr.Button("Capture Frame")

            with gr.Column():
                gr.Markdown("### Draw/Click to Annotate")
                annotator = gr.ImageEditor(
                    type = "pil",
                    brush=gr.Brush(),
                    show_label = False,
                    interactive = True,
                )

            #continuosly update the live_display with the latest frame
            demo.load(fn=live_stream, inputs=None, outputs=[live_display])  # Update every 50ms

            #when the aid clicks on the live_display, push that single frame into annotator
            capture_btn.click(fn=lambda img: img, inputs=live_display, outputs=annotator)

            def send_annotation(annotated_img):
                buffered = BytesIO()
                annotated_img.save(buffered, format="PNG")
                b64 = base64.b64encode(buffered.getvalue()).decode()
                requests.post(f"{SERVER_URL}/submit_annotation", json={"image": b64}, verify=VERIFY_SSL)
                return "Annotation sent!"

            send_btn = gr.Button("Save & Send Annotation")
            result_txt = gr.Textbox(interactive=False)
            send_btn.click(fn=send_annotation, inputs=[annotator], outputs=[result_txt])
        #gr.Markdown("<h2 style='text-align: left;'>Annotated Image w/ Obj Detection [Sent to LLM]:</h2>"
>>>>>>> d2b7fe9febc15c6eb900f232aa75779667e41bd8

        # with gr.Row():
        #     with gr.Column(scale = 1):
        #         gr.Markdown("<h2 style='text-align: left;'>Live Stream w/ Obj Detection:</h2>")
        #         image_display = gr.Image(type="pil", show_label=False)

        #         gr.Markdown("<h2 style='text-align: left;'>Annotated Image w/ Obj Detection [Sent to LLM]:</h2>")
        #         yolo_image_display = gr.Image(type="pil", show_label=False)
        #     with gr.Column(scale = 1):

        #         gr.Markdown("<h2 style='text-align: left;'>Current Image [Sent to LLM]:</h2>")
        #         user_image_display = gr.Image(type="pil", show_label=False)

        #         gr.Markdown("<h2 style='text-align: left;'>Reference Image of Completed Step [Sent to LLM]:</h2>")
        #         sample_image_display = gr.Image(type="pil", show_label=False)

        #     with gr.Column(scale = 1):
        #         gr.Markdown("<h2 style='text-align: left;'>Response from LLM:</h2>")
        #         #increase font size of textbox
        #         text_display = gr.Textbox( lines=5, max_lines=8, interactive=False, show_label = False, elem_classes="big-textbox")

        #demo.load(fn=live_stream, inputs=None, outputs=[live_display])
    demo.queue()
<<<<<<< HEAD
    demo.launch(server_name="127.0.0.1", server_port=8000, share=False)
=======
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
>>>>>>> d2b7fe9febc15c6eb900f232aa75779667e41bd8
