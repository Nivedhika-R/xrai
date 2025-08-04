# preview.py

import argparse
from io import BytesIO
import base64
import requests
import numpy as np
import cv2
from PIL import Image
import gradio as gr

# ───────────────── CLI ARGUMENTS ─────────────────
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

# ───────────────── HELPERS ─────────────────
# Create a session for connection reuse
session = requests.Session()
session.verify = VERIFY_SSL

def fetch_latest_frame():
    """Fetch the latest PIL frame from server."""
    try:
        resp = session.get(f"{SERVER_URL}/latest-frame",
                          timeout=TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        b64 = data.get("image", "")
        if not b64:
            return None
        raw = base64.b64decode(b64)
        arr = np.frombuffer(raw, np.uint8)
        cv_img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)
    except Exception as e:
        # On error, return None so Gradio shows blank or error icon
        print("Fetch error:", e)
        return None

def select_frame(frame: Image.Image):
    """Copy live frame into annotator."""
    return frame

def send_annotation(img: Image.Image):
    """POST annotated image back to your server."""
    if img is None:
        return "❌ No image to send"
    
    buf = BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    try:
        resp = session.post(f"{SERVER_URL}/submit-annotation",
                           json={"image": b64},
                           timeout=TIMEOUT)
        resp.raise_for_status()
        return "✅ Sent!"
    except Exception as e:
        return f"❌ {e}"

# ───────────────── BUILD UI ─────────────────
with gr.Blocks(css="""
    .gr-block { background: #111; color: #eee; }
    .gr-button { margin-top: 8px; }
""", 
js="""
function autoRefresh() {
    const refreshBtn = document.getElementById('refresh-btn');
    if (refreshBtn) {
        setInterval(() => {
            refreshBtn.click();
        }, 100);
    }
}
setTimeout(autoRefresh, 1000);
""") as demo:

    gr.Markdown("## XaiR Live Preview & Click‑to‑Annotate")

    with gr.Row():
        # ─ Live stream column ─
        with gr.Column():
            gr.Markdown("### Live Stream Feed")
            live_display = gr.Image(type="pil",
                                    interactive=False,
                                    show_label=False)
            with gr.Row():
                capture_btn = gr.Button("Capture Frame")
                refresh_btn = gr.Button("Refresh Stream", elem_id="refresh-btn")

        # ─ Annotation column ─
        with gr.Column():
            gr.Markdown("### Draw / Click to Annotate")
            annotator = gr.ImageEditor(
                type="pil",
                interactive=True,
                show_label=False
            )

    send_btn   = gr.Button("Save & Send Annotation")
    result_txt = gr.Textbox(label="Send Status", interactive=False)

    # ─── Initial load and manual refresh
    demo.load(fn=fetch_latest_frame,
              inputs=None,
              outputs=live_display)
    
    # ─── Manual refresh button
    refresh_btn.click(fn=fetch_latest_frame,
                     inputs=None,
                     outputs=live_display)

    # ─── On click, capture current live_display into annotator
    capture_btn.click(fn=select_frame,
                      inputs=live_display,
                      outputs=annotator)

    # ─── Send annotation back
    send_btn.click(fn=send_annotation,
                   inputs=annotator,
                   outputs=result_txt)

# ───────────────── LAUNCH ─────────────────
demo.queue()
demo.launch(server_name=args.ip,
            server_port=args.port,
            share=False)
