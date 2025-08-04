# preview_annotation.py - Updated version

import argparse
from io import BytesIO
import base64
import requests
import numpy as np
import cv2
from PIL import Image
import gradio as gr
import json

parser = argparse.ArgumentParser(description="XaiR Live Preview & Annotate")
parser.add_argument('--ip',      default='0.0.0.0', help='Host for Gradio server')
parser.add_argument('--port',   type=int, default=7860,   help='Port for Gradio server')
parser.add_argument('--backend', default='192.168.4.140',     help='Host/IP of your frame server')
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
            frame_base64 = data.get("image", "")
            if not frame_base64:
                yield None
                continue

            raw = base64.b64decode(frame_base64)
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
    if frame is None:
        return None
    return frame

def extract_coordinates_from_editor_data(editor_data):
    """
    Extract coordinates from Gradio ImageEditor data structure
    """
    coordinates = []

    if editor_data is None:
        return coordinates

    try:
        # Handle dict format from Gradio ImageEditor
        if isinstance(editor_data, dict):
            # Look for the actual image in the dict
            img = None
            if 'image' in editor_data:
                img = editor_data['image']
            elif 'background' in editor_data:
                img = editor_data['background']
            elif 'composite' in editor_data:
                img = editor_data['composite']
            else:
                # Find any PIL Image in the dict
                for value in editor_data.values():
                    if hasattr(value, 'save'):  # PIL Image check
                        img = value
                        break

            if img is not None:
                coordinates = analyze_image_for_annotations(img)
        else:
            # Direct PIL Image
            coordinates = analyze_image_for_annotations(editor_data)

    except Exception as e:
        print(f"Error extracting coordinates: {e}")
        return []

    return coordinates

def analyze_image_for_annotations(image):
    """
    Analyze PIL image for drawn annotations
    """
    coordinates = []

    try:
        # Convert PIL to numpy array
        img_array = np.array(image)

        # Look for drawn lines/annotations
        # Convert to grayscale for edge detection
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array

        # Use edge detection to find drawn lines
        edges = cv2.Canny(gray, 50, 150)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            # Skip very small contours
            if cv2.contourArea(contour) < 20:
                continue

            # Get points along the contour
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            for point in approx:
                x, y = point[0]
                coordinates.append([int(x), int(y)])

        # Alternative: detect specific annotation colors
        if not coordinates:
            coordinates = detect_annotation_colors(img_array)

    except Exception as e:
        print(f"Error analyzing image: {e}")
        return []

    return coordinates

def detect_annotation_colors(img_array):
    """
    Detect common annotation colors
    """
    coordinates = []

    # Common drawing colors in Gradio ImageEditor
    colors_to_detect = [
        ([255, 0, 0], "red"),       # Red
        ([0, 255, 0], "green"),     # Green
        ([0, 0, 255], "blue"),      # Blue
        ([255, 255, 0], "yellow"),  # Yellow
        ([255, 0, 255], "magenta"), # Magenta
        ([0, 255, 255], "cyan"),    # Cyan
        ([0, 0, 0], "black"),       # Black
        ([255, 255, 255], "white")  # White
    ]

    for color, name in colors_to_detect:
        # Create mask for this color (with tolerance)
        if len(img_array.shape) == 3:
            lower = np.array([max(0, c-20) for c in color])
            upper = np.array([min(255, c+20) for c in color])
            mask = cv2.inRange(img_array, lower, upper)
        else:
            # Grayscale
            if name in ['black']:
                mask = (img_array < 30).astype(np.uint8) * 255
            elif name in ['white']:
                mask = (img_array > 225).astype(np.uint8) * 255
            else:
                continue

        # Find coordinates of this color
        y_coords, x_coords = np.where(mask > 0)

        if len(x_coords) > 10:  # Only if we found enough pixels
            # Cluster points to avoid too many coordinates
            points = list(zip(x_coords, y_coords))
            # Sample every 10th point to reduce noise
            sampled_points = points[::10]
            coordinates.extend([[int(x), int(y)] for x, y in sampled_points])

    return coordinates

def send_annotation(editor_data):
    """POST annotated image and coordinates back to your server."""
    try:
        print(f"Editor data type: {type(editor_data)}")
        print(f"Editor data keys: {editor_data.keys() if isinstance(editor_data, dict) else 'Not a dict'}")

        # Extract coordinates from the editor data
        coordinates = extract_coordinates_from_editor_data(editor_data)

        # Handle different formats that Gradio ImageEditor might return
        if isinstance(editor_data, dict):
            # Check common keys that Gradio might use
            if 'image' in editor_data:
                img = editor_data['image']
            elif 'background' in editor_data:
                img = editor_data['background']
            elif 'composite' in editor_data:
                img = editor_data['composite']
            else:
                # Try to find any PIL Image in the dict values
                img = None
                for key, value in editor_data.items():
                    if hasattr(value, 'save'):  # Check if it's a PIL Image
                        img = value
                        print(f"Found image in key: {key}")
                        break
                if img is None:
                    return f"No image found in data. Available keys: {list(editor_data.keys())}"
        else:
            # It's directly a PIL Image
            img = editor_data

        if img is None:
            return "No image to send"

        # Convert PIL image to base64
        buf = BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()

        # Send both image and coordinates
        payload = {
            "image": b64,
            "coordinates": coordinates,
            "annotation_count": len(coordinates)
        }

        resp = requests.post(f"{SERVER_URL}/submit-annotation",
                             json=payload,
                             verify=VERIFY_SSL,
                             timeout=TIMEOUT)
        resp.raise_for_status()

        result = resp.json()
        return f"Sent! Found {len(coordinates)} annotation points. Server response: {result.get('message', 'OK')}"

    except Exception as e:
        print(f"Full error details: {e}")
        import traceback
        traceback.print_exc()
        return f"Error: {e}"

# UI part
with gr.Blocks(css="""
    .gr-block { background: #111; color: #eee; }
    .gr-button { margin-top: 8px; }
    #send-annotation-btn button { width:auto; padding: 6px 12px; font-size: 14px; max-width: 120px; }
    #capture-btn button { width:auto; padding: 6px 12px; font-size: 14px; max-width: 100px; }
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
                type="pil",
                interactive=True,
                show_label=False,
                height=480,
                width=640,
                brush=gr.Brush(default_size=3, colors=["#ff0000", "#00ff00", "#0000ff", "#ffff00"])
            )

    send_btn   = gr.Button("Extract & Send Coordinates", elem_id="send-annotation-btn")
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
