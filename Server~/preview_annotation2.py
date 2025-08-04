# # preview_annotation.py - Updated version

# import argparse
# from io import BytesIO
# import base64
# import requests
# import numpy as np
# import cv2
# from PIL import Image
# import gradio as gr
# # import json

# parser = argparse.ArgumentParser(description="XaiR Live Preview & Annotate")
# parser.add_argument('--ip',      default='0.0.0.0', help='Host for Gradio server')
# parser.add_argument('--port',   type=int, default=7860,   help='Port for Gradio server')
# parser.add_argument('--backend', default='127.0.0.1',     help='Host/IP of your frame server')
# parser.add_argument('--bport',  type=int, default=8000,   help='Port of your frame server')
# parser.add_argument('--https', action='store_true',       help='Use HTTPS for backend')
# args = parser.parse_args()

# PROTO      = 'https' if args.https else 'http'
# SERVER_URL = f"{PROTO}://{args.backend}:{args.bport}"
# VERIFY_SSL = False      # True if you have a valid cert
# TIMEOUT    = 5          # seconds

# # def latest_frame():
# #     """Generator that yields the latest PIL frame indefinitely."""
# #     while True:
# #         try:
# #             resp = requests.get(f"{SERVER_URL}/latest-frame",
# #                                 timeout=TIMEOUT,
# #                                 verify=VERIFY_SSL)
# #             resp.raise_for_status()
# #             data = resp.json()
# #             b64 = data.get("image", "")
# #             if not b64:
# #                 yield None
# #                 continue

# #             raw = base64.b64decode(b64)
# #             arr = np.frombuffer(raw, np.uint8)
# #             cv_img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
# #             rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)

# #             #convert to PIL and rotate 90 deg clockwise
# #             pil = Image.fromarray(rgb)
# #             pil = pil.rotate(-90, expand=True)
# #             yield pil
# #         except Exception as e:
# #             # On error, yield None so Gradio shows blank or error icon
# #             print("Fetch error:", e)
# #             yield None

# # Add this right after the args parsing
# print("=" * 50)
# print("🔧 GRADIO CONFIGURATION DEBUG")
# print("=" * 50)
# print(f"Backend IP: {args.backend}")
# print(f"Backend Port: {args.bport}")
# print(f"HTTPS: {args.https}")
# print(f"SERVER_URL: {SERVER_URL}")
# print(f"Full latest-frame URL: {SERVER_URL}/latest-frame")
# print("=" * 50)

# def fetch_latest_frame():
#     import time
#     frame_count = 0
#     while True:
#         start_time = time.time()
#         frame_count += 1
#         try:
#             url = f"{SERVER_URL}/latest-frame?ts={time.time()}"
#             resp = requests.get(url, timeout=TIMEOUT, verify=VERIFY_SSL)
#             resp.raise_for_status()
#             data = resp.json()
#             b64 = data.get("image", "")

#             if not b64:
#                 yield None
#                 continue

#             raw = base64.b64decode(b64)
#             arr = np.frombuffer(raw, np.uint8)
#             cv_img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
#             rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)

#             pil = Image.fromarray(rgb).rotate(-90, expand=True).copy()
#             yield pil

#         except Exception as e:
#             print(f"❌ Frame #{frame_count} fetch error: {e}")
#             yield None

#         # Adaptive delay to ~10 FPS
#         elapsed = time.time() - start_time
#         if elapsed < 0.1:
#             time.sleep(0.1 - elapsed)


# def select_frame(frame: Image.Image):
#     """Copy live frame into annotator."""
#     if frame is None:
#         return None
#     return frame

# def extract_coordinates_from_editor_data(editor_data):
#     """
#     Extract coordinates from Gradio ImageEditor data structure
#     """
#     coordinates = []

#     if editor_data is None:
#         return coordinates

#     try:
#         # Handle dict format from Gradio ImageEditor
#         if isinstance(editor_data, dict):
#             # Look for the actual image in the dict
#             img = None
#             if 'image' in editor_data:
#                 img = editor_data['image']
#             elif 'background' in editor_data:
#                 img = editor_data['background']
#             elif 'composite' in editor_data:
#                 img = editor_data['composite']
#             else:
#                 # Find any PIL Image in the dict
#                 for value in editor_data.values():
#                     if hasattr(value, 'save'):  # PIL Image check
#                         img = value
#                         break

#             if img is not None:
#                 coordinates = analyze_image_for_annotations(img)
#         else:
#             # Direct PIL Image
#             coordinates = analyze_image_for_annotations(editor_data)

#     except Exception as e:
#         print(f"Error extracting coordinates: {e}")
#         return []

#     return coordinates

# def analyze_image_for_annotations(image):
#     """
#     Analyze PIL image for drawn annotations
#     """
#     coordinates = []

#     try:
#         # Convert PIL to numpy array
#         img_array = np.array(image)

#         # Look for drawn lines/annotations
#         # Convert to grayscale for edge detection
#         if len(img_array.shape) == 3:
#             gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
#         else:
#             gray = img_array

#         # Use edge detection to find drawn lines
#         edges = cv2.Canny(gray, 50, 150)

#         # Find contours
#         contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#         for contour in contours:
#             # Skip very small contours
#             if cv2.contourArea(contour) < 20:
#                 continue

#             # Get points along the contour
#             epsilon = 0.02 * cv2.arcLength(contour, True)
#             approx = cv2.approxPolyDP(contour, epsilon, True)

#             for point in approx:
#                 x, y = point[0]
#                 coordinates.append([int(x), int(y)])

#         # Alternative: detect specific annotation colors
#         if not coordinates:
#             coordinates = detect_annotation_colors(img_array)

#     except Exception as e:
#         print(f"Error analyzing image: {e}")
#         return []

#     return coordinates

# def detect_annotation_colors(img_array):
#     """
#     Detect common annotation colors
#     """
#     coordinates = []

#     # Common drawing colors in Gradio ImageEditor
#     colors_to_detect = [
#         ([255, 0, 0], "red"),       # Red
#         ([0, 255, 0], "green"),     # Green
#         ([0, 0, 255], "blue"),      # Blue
#         ([255, 255, 0], "yellow"),  # Yellow
#         ([255, 0, 255], "magenta"), # Magenta
#         ([0, 255, 255], "cyan"),    # Cyan
#         ([0, 0, 0], "black"),       # Black
#         ([255, 255, 255], "white")  # White
#     ]

#     for color, name in colors_to_detect:
#         # Create mask for this color (with tolerance)
#         if len(img_array.shape) == 3:
#             lower = np.array([max(0, c-20) for c in color])
#             upper = np.array([min(255, c+20) for c in color])
#             mask = cv2.inRange(img_array, lower, upper)
#         else:
#             # Grayscale
#             if name in ['black']:
#                 mask = (img_array < 30).astype(np.uint8) * 255
#             elif name in ['white']:
#                 mask = (img_array > 225).astype(np.uint8) * 255
#             else:
#                 continue

#         # Find coordinates of this color
#         y_coords, x_coords = np.where(mask > 0)

#         if len(x_coords) > 10:  # Only if we found enough pixels
#             # Cluster points to avoid too many coordinates
#             points = list(zip(x_coords, y_coords))
#             # Sample every 10th point to reduce noise
#             sampled_points = points[::10]
#             coordinates.extend([[int(x), int(y)] for x, y in sampled_points])

#     return coordinates

# def send_annotation(editor_data):
#     """POST annotated image and coordinates back to your server."""
#     try:
#         print(f"Editor data type: {type(editor_data)}")
#         print(f"Editor data keys: {editor_data.keys() if isinstance(editor_data, dict) else 'Not a dict'}")

#         # Extract coordinates from the editor data
#         coordinates = extract_coordinates_from_editor_data(editor_data)

#         # Handle different formats that Gradio ImageEditor might return
#         if isinstance(editor_data, dict):
#             # Check common keys that Gradio might use
#             if 'image' in editor_data:
#                 img = editor_data['image']
#             elif 'background' in editor_data:
#                 img = editor_data['background']
#             elif 'composite' in editor_data:
#                 img = editor_data['composite']
#             else:
#                 # Try to find any PIL Image in the dict values
#                 img = None
#                 for key, value in editor_data.items():
#                     if hasattr(value, 'save'):  # Check if it's a PIL Image
#                         img = value
#                         print(f"Found image in key: {key}")
#                         break
#                 if img is None:
#                     return f"No image found in data. Available keys: {list(editor_data.keys())}"
#         else:
#             # It's directly a PIL Image
#             img = editor_data

#         if img is None:
#             return "No image to send"

#         # Convert PIL image to base64
#         buf = BytesIO()
#         img.save(buf, format="PNG")
#         b64 = base64.b64encode(buf.getvalue()).decode()

#         # Send both image and coordinates
#         payload = {
#             "image": b64,
#             "coordinates": coordinates,
#             "annotation_count": len(coordinates)
#         }

#         resp = requests.post(f"{SERVER_URL}/submit-annotation",
#                              json=payload,
#                              verify=VERIFY_SSL,
#                              timeout=TIMEOUT)
#         resp.raise_for_status()

#         result = resp.json()
#         return f"Sent! Found {len(coordinates)} annotation points. Server response: {result.get('message', 'OK')}"
    
    

#     except Exception as e:
#         print(f"Full error details: {e}")
#         import traceback
#         traceback.print_exc()
#         return f"Error: {e}"
    
# print("🧪 Testing server connection...")
# try:
#     test_resp = requests.get(f"{SERVER_URL}/latest-frame", timeout=5, verify=VERIFY_SSL)
#     print(f"✅ Server test successful: {test_resp.status_code}")
#     test_data = test_resp.json()
#     print(f"Has image: {test_data.get('image') is not None}")
#     if test_data.get('image'):
#         print(f"Image length: {len(test_data['image'])}")
# except Exception as e:
#     print(f"❌ Server test failed: {e}")
# print("-" * 30)

# # UI part
# with gr.Blocks(css="""
#     .gr-block { background: #111; color: #eee; }
#     .gr-button { margin-top: 8px; }
#     #send-annotation-btn button { width:auto; padding: 6px 12px; font-size: 14px; max-width: 120px; }
#     #capture-btn button { width:auto; padding: 6px 12px; font-size: 14px; max-width: 100px; }
# """) as demo:

#     gr.Markdown("## XaiR Live Preview & Click‑to‑Annotate")

#     with gr.Row():
#         # ─ Live stream column ─
#         with gr.Column():
#             gr.Markdown("### Live Stream Feed")
#             live_display = gr.Image(type="pil",
#                                     interactive=False,
#                                     show_label=False,
#                                     height=480,
#                                     width=640)
#             capture_btn  = gr.Button("Capture Frame", elem_id="capture-btn")

#         # ─ Annotation column ─
#         with gr.Column():
#             gr.Markdown("### Draw / Click to Annotate")
#             annotator = gr.ImageEditor(
#                 type="pil",
#                 interactive=True,
#                 show_label=False,
#                 height=480,
#                 width=640,
#                 brush=gr.Brush(default_size=3, colors=["#ff0000", "#00ff00", "#0000ff", "#ffff00"])
#             )

#     send_btn   = gr.Button("Extract & Send Coordinates", elem_id="send-annotation-btn")
#     result_txt = gr.Textbox(label="Send Status", interactive=False)

#     # Continuous live stream into live_display
#     demo.load(fn=fetch_latest_frame,
#               inputs=None,
#               outputs=live_display)

#     # On click, capture current live_display into annotator
#     capture_btn.click(fn=select_frame,
#                       inputs=live_display,
#                       outputs=annotator)

#     # Send annotation back
#     send_btn.click(fn=send_annotation,
#                    inputs=annotator,
#                    outputs=result_txt)

# # launch the Gradio app
# demo.queue()
# demo.launch(server_name=args.ip,
#             server_port=args.port,
#             share=False)

# preview_annotation_fixed.py - Fixed version

import argparse
from io import BytesIO
import base64
import requests
import numpy as np
import cv2
from PIL import Image
import gradio as gr
import time
import urllib3

# Disable SSL warnings for self-signed certificates
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

parser = argparse.ArgumentParser(description="XaiR Live Preview & Annotate")
parser.add_argument('--ip',      default='0.0.0.0', help='Host for Gradio server')
parser.add_argument('--port',   type=int, default=7860,   help='Port for Gradio server')
parser.add_argument('--backend', default='127.0.0.1',     help='Host/IP of your frame server')
parser.add_argument('--bport',  type=int, default=8000,   help='Port of your frame server')
parser.add_argument('--https', action='store_true',       help='Use HTTPS for backend')
args = parser.parse_args()

PROTO      = 'https' if args.https else 'http'
SERVER_URL = f"{PROTO}://{args.backend}:{args.bport}"
VERIFY_SSL = False      # Keep False for self-signed certs
TIMEOUT    = 10         # Increased timeout

print("=" * 50)
print("🔧 GRADIO CONFIGURATION DEBUG")
print("=" * 50)
print(f"Backend IP: {args.backend}")
print(f"Backend Port: {args.bport}")
print(f"HTTPS: {args.https}")
print(f"SERVER_URL: {SERVER_URL}")
print(f"Full latest-frame URL: {SERVER_URL}/latest-frame")
print("=" * 50)

# Create a session with proper configuration
session = requests.Session()
session.verify = VERIFY_SSL
session.timeout = TIMEOUT

# Add retry logic and better error handling
def fetch_latest_frame():
    import time
    frame_count = 0
    consecutive_errors = 0
    max_consecutive_errors = 5
    last_successful_frame = None
    
    while True:
        start_time = time.time()
        frame_count += 1
        
        try:
            # Add cache-busting parameter and proper headers
            url = f"{SERVER_URL}/latest-frame?ts={time.time()}"
            headers = {
                'Cache-Control': 'no-cache, no-store, must-revalidate',
                'Pragma': 'no-cache',
                'Expires': '0',
                'Connection': 'keep-alive'
            }
            
            print(f"🔄 Fetching frame #{frame_count} from {url}")
            
            # Use a fresh request each time instead of session
            resp = requests.get(url, headers=headers, verify=VERIFY_SSL, timeout=5)
            resp.raise_for_status()
            
            data = resp.json()
            
            # Check for valid response
            status = data.get("status", "unknown")
            if status == "no_frames":
                print(f"⚠️  Frame #{frame_count}: No frames available yet")
                # Return last successful frame if available
                if last_successful_frame is not None:
                    yield last_successful_frame
                else:
                    yield None
                time.sleep(0.5)  # Wait longer when no frames
                continue
                
            b64 = data.get("image", "")
            if not b64:
                print(f"⚠️  Frame #{frame_count}: Empty image data")
                if last_successful_frame is not None:
                    yield last_successful_frame
                else:
                    yield None
                continue

            # Successfully got image data
            consecutive_errors = 0
            
            try:
                raw = base64.b64decode(b64)
                arr = np.frombuffer(raw, np.uint8)
                cv_img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                
                if cv_img is None:
                    print(f"❌ Frame #{frame_count}: Failed to decode image")
                    yield last_successful_frame if last_successful_frame is not None else None
                    continue
                    
                rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                pil = Image.fromarray(rgb).rotate(-90, expand=True).copy()
                
                # Store as last successful frame
                last_successful_frame = pil
                
                print(f"✅ Frame #{frame_count}: Successfully processed ({pil.size[0]}x{pil.size[1]})")
                yield pil
                
            except Exception as decode_error:
                print(f"❌ Frame #{frame_count} decode error: {decode_error}")
                yield last_successful_frame if last_successful_frame is not None else None

        except requests.exceptions.ConnectionError as e:
            consecutive_errors += 1
            print(f"❌ Frame #{frame_count} connection error (attempt {consecutive_errors}/{max_consecutive_errors})")
            if consecutive_errors >= max_consecutive_errors:
                print("❌ Too many connection errors, returning last frame")
                yield last_successful_frame if last_successful_frame is not None else None
                time.sleep(2)  # Wait longer for server to recover
                consecutive_errors = 0
            else:
                yield last_successful_frame if last_successful_frame is not None else None
                
        except requests.exceptions.Timeout:
            print(f"⏰ Frame #{frame_count} timeout")
            yield last_successful_frame if last_successful_frame is not None else None
            
        except Exception as e:
            print(f"❌ Frame #{frame_count} unexpected error: {type(e).__name__}: {e}")
            yield last_successful_frame if last_successful_frame is not None else None

        # Adaptive delay targeting ~10 FPS
        elapsed = time.time() - start_time
        target_delay = 0.1  # 100ms for 10 FPS
        
        # Increase delay if we're getting errors
        if consecutive_errors > 0:
            target_delay = 0.5  # Slow down when errors occur
            
        if elapsed < target_delay:
            time.sleep(target_delay - elapsed)
            
def select_frame(frame: Image.Image):
    """Copy live frame into annotator."""
    if frame is None:
        print("⚠️  Cannot select frame: frame is None")
        return None
    print(f"📸 Frame selected for annotation: {frame.size[0]}x{frame.size[1]}")
    return frame.copy()

def extract_coordinates_from_editor_data(editor_data):
    """Extract coordinates from Gradio ImageEditor data structure"""
    coordinates = []

    if editor_data is None:
        print("⚠️  Editor data is None")
        return coordinates

    try:
        print(f"🔍 Editor data type: {type(editor_data)}")
        
        # Handle dict format from Gradio ImageEditor
        if isinstance(editor_data, dict):
            print(f"📋 Editor data keys: {list(editor_data.keys())}")
            
            # Look for the actual image in the dict
            img = None
            for key in ['image', 'composite', 'background', 'layers']:
                if key in editor_data and editor_data[key] is not None:
                    img = editor_data[key]
                    print(f"🖼️  Found image in key: {key}")
                    break
            
            if img is None:
                # Find any PIL Image in the dict
                for key, value in editor_data.items():
                    if hasattr(value, 'save'):  # PIL Image check
                        img = value
                        print(f"🖼️  Found PIL image in key: {key}")
                        break

            if img is not None:
                coordinates = analyze_image_for_annotations(img)
            else:
                print("❌ No image found in editor data")
        else:
            # Direct PIL Image
            print("🖼️  Direct PIL image detected")
            coordinates = analyze_image_for_annotations(editor_data)

    except Exception as e:
        print(f"❌ Error extracting coordinates: {e}")
        import traceback
        traceback.print_exc()
        return []

    print(f"📍 Extracted {len(coordinates)} coordinate points")
    return coordinates

def analyze_image_for_annotations(image):
    """Analyze PIL image for drawn annotations"""
    coordinates = []

    try:
        print(f"🔍 Analyzing image: {image.size[0]}x{image.size[1]}")
        
        # Convert PIL to numpy array
        img_array = np.array(image)
        print(f"📊 Image array shape: {img_array.shape}")

        # Look for drawn lines/annotations
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array

        # Use edge detection to find drawn lines
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(f"🔍 Found {len(contours)} contours")

        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            # Skip very small contours
            if area < 20:
                continue
                
            print(f"📐 Contour {i}: area = {area}")

            # Get points along the contour
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            for point in approx:
                x, y = point[0]
                coordinates.append([int(x), int(y)])

        # Alternative: detect specific annotation colors if no contours found
        if not coordinates:
            print("🎨 No contours found, trying color detection...")
            coordinates = detect_annotation_colors(img_array)

    except Exception as e:
        print(f"❌ Error analyzing image: {e}")
        import traceback
        traceback.print_exc()
        return []

    return coordinates

def detect_annotation_colors(img_array):
    """Detect common annotation colors"""
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
    ]

    for color, name in colors_to_detect:
        try:
            # Create mask for this color (with tolerance)
            if len(img_array.shape) == 3:
                lower = np.array([max(0, c-30) for c in color])  # Increased tolerance
                upper = np.array([min(255, c+30) for c in color])
                mask = cv2.inRange(img_array, lower, upper)
            else:
                # Grayscale
                if name == 'black':
                    mask = (img_array < 50).astype(np.uint8) * 255
                else:
                    continue

            # Find coordinates of this color
            y_coords, x_coords = np.where(mask > 0)

            if len(x_coords) > 10:  # Only if we found enough pixels
                print(f"🎨 Found {len(x_coords)} pixels of {name} color")
                # Cluster points to avoid too many coordinates
                points = list(zip(x_coords, y_coords))
                # Sample every 5th point to get more detail but avoid too much noise
                sampled_points = points[::5]
                coordinates.extend([[int(x), int(y)] for x, y in sampled_points])
                
        except Exception as e:
            print(f"❌ Error detecting {name} color: {e}")
            continue

    return coordinates

def send_annotation(editor_data):
    """POST annotated image and coordinates back to your server."""
    try:
        print("=" * 40)
        print("📤 SENDING ANNOTATION")
        print("=" * 40)
        print(f"Editor data type: {type(editor_data)}")
        
        if isinstance(editor_data, dict):
            print(f"Editor data keys: {list(editor_data.keys())}")

        # Extract coordinates from the editor data
        coordinates = extract_coordinates_from_editor_data(editor_data)

        # Handle different formats that Gradio ImageEditor might return
        img = None
        if isinstance(editor_data, dict):
            # Check common keys that Gradio might use
            for key in ['image', 'composite', 'background', 'layers']:
                if key in editor_data and editor_data[key] is not None:
                    img = editor_data[key]
                    print(f"🖼️  Using image from key: {key}")
                    break
                    
            if img is None:
                # Try to find any PIL Image in the dict values
                for key, value in editor_data.items():
                    if hasattr(value, 'save'):  # Check if it's a PIL Image
                        img = value
                        print(f"🖼️  Found PIL image in key: {key}")
                        break
                        
            if img is None:
                return f"❌ No image found in data. Available keys: {list(editor_data.keys())}"
        else:
            # It's directly a PIL Image
            img = editor_data
            print("🖼️  Using direct PIL image")

        if img is None:
            return "❌ No image to send"

        print(f"📏 Image size: {img.size[0]}x{img.size[1]}")
        print(f"📍 Found {len(coordinates)} annotation points")

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

        print(f"📤 Sending to: {SERVER_URL}/submit-annotation")
        
        resp = session.post(f"{SERVER_URL}/submit-annotation",
                           json=payload,
                           verify=VERIFY_SSL,
                           timeout=TIMEOUT)
        resp.raise_for_status()

        result = resp.json()
        success_msg = f"✅ Sent! Found {len(coordinates)} annotation points. Server response: {result.get('message', 'OK')}"
        print(success_msg)
        return success_msg
    
    except Exception as e:
        error_msg = f"❌ Error: {e}"
        print(f"❌ Full error details: {e}")
        import traceback
        traceback.print_exc()
        return error_msg

# Test server connection
print("🧪 Testing server connection...")
try:
    test_resp = session.get(f"{SERVER_URL}/latest-frame", verify=VERIFY_SSL, timeout=TIMEOUT)
    print(f"✅ Server test successful: {test_resp.status_code}")
    test_data = test_resp.json()
    has_image = test_data.get('image') is not None
    print(f"Has image: {has_image}")
    if has_image:
        print(f"Image data length: {len(test_data['image'])}")
except Exception as e:
    print(f"❌ Server test failed: {e}")
    print("🔧 Make sure your server is running and accessible")
print("-" * 50)

# UI part
with gr.Blocks(css="""
    .gr-block { background: #111; color: #eee; }
    .gr-button { margin-top: 8px; }
    #send-annotation-btn button { width:auto; padding: 6px 12px; font-size: 14px; max-width: 150px; }
    #capture-btn button { width:auto; padding: 6px 12px; font-size: 14px; max-width: 120px; }
    .status-text { font-family: monospace; background: #222; padding: 10px; }
""") as demo:

    gr.Markdown("## XaiR Live Preview & Click‑to‑Annotate")
    gr.Markdown("### 🔴 Live camera feed should appear below. Click 'Capture Frame' to annotate it.")

    with gr.Row():
        # ─ Live stream column ─
        with gr.Column():
            gr.Markdown("### 📹 Live Stream Feed")
            live_display = gr.Image(type="pil",
                                    interactive=False,
                                    show_label=False,
                                    height=480,
                                    width=640)
            capture_btn = gr.Button("📸 Capture Frame", elem_id="capture-btn")

        # ─ Annotation column ─
        with gr.Column():
            gr.Markdown("### ✏️ Draw / Click to Annotate")
            annotator = gr.ImageEditor(
                type="pil",
                interactive=True,
                show_label=False,
                height=480,
                width=640,
                brush=gr.Brush(default_size=3, colors=["#ff0000", "#00ff00", "#0000ff", "#ffff00"])
            )

    send_btn = gr.Button("📤 Extract & Send Coordinates", elem_id="send-annotation-btn")
    result_txt = gr.Textbox(label="📊 Send Status", interactive=False, elem_classes=["status-text"])

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

if __name__ == "__main__":
    # Launch the Gradio app
    demo.queue(max_size=20)  # Limit queue size
    demo.launch(server_name=args.ip,
                server_port=args.port,
                share=False,
                show_error=True)
