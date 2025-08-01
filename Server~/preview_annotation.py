# preview.py

import argparse
from io import BytesIO
import base64
import requests
import numpy as np
import cv2
from PIL import Image
import gradio as gr

parser = argparse.ArgumentParser(description="XaiR Live Preview & Annotate")
parser.add_argument('--ip',      default='0.0.0.0', help='Host for Gradio server')
parser.add_argument('--port',   type=int, default=7860,   help='Port for Gradio server')
parser.add_argument('--backend', default='127.0.0.1',     help='Host/IP of your frame server')
parser.add_argument('--bport',  type=int, default=8000,   help='Port of your frame server')
parser.add_argument('--https', action='store_true',       help='Use HTTPS for backend')
args = parser.parse_args()

PROTO      = 'https' if args.https else 'http'
SERVER_URL = f"{PROTO}://{args.backend}:{args.bport}"
VERIFY_SSL = False      # True if you have a valid cert
TIMEOUT    = 5          # seconds

def fetch_latest_frame():
    """Generator that yields the latest PIL frame indefinitely."""
    while True:
        try:
            resp = requests.get(f"{SERVER_URL}/latest-frame",
                                timeout=TIMEOUT,
                                verify=VERIFY_SSL)
            resp.raise_for_status()
            data = resp.json()
            b64 = data.get("image", "")
            raw = base64.b64decode(b64)
            arr = np.frombuffer(raw, np.uint8)
            cv_img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)

            #convert to PIL and rotate 90 deg clockwise
            pil = Image.fromarray(rgb)
            pil = pil.rotate(-90, expand=True)
            yield pil
        except Exception as e:
            # On error, yield None so Gradio shows blank or error icon
            print("Fetch error:", e)
            yield None

def select_frame(frame: Image.Image):
    """Copy live frame into annotator."""
    return frame

def send_annotation(img: Image.Image):
    """POST annotated image back to your server."""
    buf = BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    try:
        resp = requests.post(f"{SERVER_URL}/submit-annotation",
                             json={"image": b64},
                             verify=VERIFY_SSL,
                             timeout=TIMEOUT)
        resp.raise_for_status()
        return "Sent!"
    except Exception as e:
        return f"{e}"

# ui part
with gr.Blocks(css="""
    .gr-block { background: #111; color: #eee; }
    .gr-button { margin-top: 8px; }
    #send-annotation-btn button { width:auto; padding: 6px 12px; font-size: 14px; max-width: 50px; }
    #capture-btn button { width:auto; padding: 6px 12px; font-size: 14px; max-width: 60px; }
""") as demo:

    gr.Markdown("## XaiR Live Preview & Click‑to‑Annotate")

    with gr.Row():
        # ─ Live stream column ─
        with gr.Column():
            gr.Markdown("### Live Stream Feed")
            live_display = gr.Image(type="pil",
                                    interactive=False,
                                    show_label=False,
                                    height=480,
                                    width=640)
            capture_btn  = gr.Button("Capture Frame", elem_id="capture-btn")

        # ─ Annotation column ─
        with gr.Column():
            gr.Markdown("### Draw / Click to Annotate")
            annotator = gr.ImageEditor(
                brush=gr.Brush(),
                type="pil",
                interactive=True,
                show_label=False,
                height=480,
                width=640
            )

    send_btn   = gr.Button("Save & Send Annotation",elem_id="send-annotation-btn")
    result_txt = gr.Textbox(label="Send Status", interactive=False)

    # Continuous live stream into live_display
    demo.load(fn=fetch_latest_frame,
              inputs=None,
              outputs=live_display)

    # On click, capture current live_display into annotator
    capture_btn.click(fn=select_frame,
                      inputs=live_display,
                      outputs=annotator)

    # Send annotation back
    send_btn.click(fn=send_annotation,
                   inputs=annotator,
                   outputs=result_txt)

# launch the Gradio app
demo.queue()
demo.launch(server_name=args.ip,
            server_port=args.port,
            share=False)
