
import argparse
from io import BytesIO
import base64
import requests
import numpy as np
import cv2
from PIL import Image
import gradio as gr
import time
import threading
import queue
import urllib3

# Disable SSL warnings for self-signed certificates
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

parser = argparse.ArgumentParser(description="XaiR Live Preview & Annotate")
parser.add_argument('--ip',      default='0.0.0.0', help='Host for Gradio server')
parser.add_argument('--port',   type=int, default=7860,   help='Port for Gradio server')
parser.add_argument('--backend', default='127.0.0.1',     help='Host/IP of your frame server')
parser.add_argument('--bport',  type=int, default=8000,   help='Port of your frame server')
parser.add_argument('--http', action='store_true',       help='Use HTTP for backend')
args = parser.parse_args()

PROTO      = 'http' if args.http else 'https'
SERVER_URL = f"{PROTO}://{args.backend}:{args.bport}"
VERIFY_SSL = False
TIMEOUT    = 8

print("=" * 60)
print("🔧 GRADIO LIVE STREAMING CONFIGURATION")
print("=" * 60)
print(f"Backend IP: {args.backend}")
print(f"Backend Port: {args.bport}")
print(f"HTTP: {args.http}")
print(f"SERVER_URL: {SERVER_URL}")
print(f"Full latest-frame URL: {SERVER_URL}/latest-frame")
print("=" * 60)

# Global state for live streaming
frame_queue = queue.Queue(maxsize=5)
streaming_active = False
current_frame = None
stream_stats = {
    'total_frames': 0,
    'successful_frames': 0,
    'errors': 0,
    'last_update': time.time()
}

def background_frame_fetcher():
    """Background thread that continuously fetches frames"""
    global streaming_active, current_frame, stream_stats

    consecutive_errors = 0
    max_errors = 10

    while streaming_active:
        try:
            url = f"{SERVER_URL}/latest-frame?ts={time.time()}"
            headers = {
                'Cache-Control': 'no-cache, no-store, must-revalidate',
                'Pragma': 'no-cache'
            }

            resp = requests.get(url, headers=headers, verify=VERIFY_SSL, timeout=5)
            resp.raise_for_status()

            data = resp.json()
            stream_stats['total_frames'] += 1

            status = data.get("status", "unknown")
            b64_image = data.get("image", "")

            if status == "success" and b64_image:
                try:
                    # Decode and process image
                    raw = base64.b64decode(b64_image)
                    arr = np.frombuffer(raw, np.uint8)
                    cv_img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

                    if cv_img is not None:
                        rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                        pil_image = Image.fromarray(rgb).rotate(-90, expand=True)

                        # Update current frame (thread-safe)
                        current_frame = pil_image.copy()
                        stream_stats['successful_frames'] += 1
                        stream_stats['last_update'] = time.time()
                        consecutive_errors = 0

                        print(f"✅ Frame {stream_stats['successful_frames']}: {pil_image.size[0]}x{pil_image.size[1]}")

                except Exception as decode_error:
                    print(f"❌ Decode error: {decode_error}")
                    stream_stats['errors'] += 1
                    consecutive_errors += 1
            else:
                print(f"⚠️  No frame available: {status}")

        except requests.exceptions.ConnectionError:
            consecutive_errors += 1
            stream_stats['errors'] += 1
            print(f"❌ Connection error (#{consecutive_errors})")
            if consecutive_errors >= max_errors:
                print("❌ Too many errors, slowing down...")
                time.sleep(2)
                consecutive_errors = 0

        except Exception as e:
            consecutive_errors += 1
            stream_stats['errors'] += 1
            print(f"❌ Fetch error: {e}")

        # Adaptive delay based on success rate
        if consecutive_errors == 0:
            time.sleep(0.1)  # 10 FPS when working
        else:
            time.sleep(0.5)  # Slower when errors

def get_current_frame():
    """Get the current frame for Gradio display"""
    global current_frame, stream_stats

    if current_frame is None:
        # Return a placeholder image
        placeholder = Image.new('RGB', (640, 480), color='black')
        return placeholder

    return current_frame.copy()

def get_stream_status():
    """Get streaming status for display"""
    global stream_stats

    if stream_stats['successful_frames'] == 0:
        return "🔴 No frames received yet"

    time_since_last = time.time() - stream_stats['last_update']

    if time_since_last > 5:
        status = "🟡 Stream paused"
    else:
        status = "🟢 Live streaming"

    return f"{status} | Frames: {stream_stats['successful_frames']}/{stream_stats['total_frames']} | Errors: {stream_stats['errors']}"

def start_streaming():
    """Start the background streaming thread"""
    global streaming_active

    if not streaming_active:
        streaming_active = True
        thread = threading.Thread(target=background_frame_fetcher, daemon=True)
        thread.start()
        print("🚀 Started background frame fetcher")
        return "🟢 Streaming started"
    return "⚠️  Already streaming"

def stop_streaming():
    """Stop the background streaming"""
    global streaming_active
    streaming_active = False
    print("🛑 Stopped background frame fetcher")
    return "🔴 Streaming stopped"

def select_frame(frame: Image.Image):
    """Copy live frame into annotator"""
    if frame is None:
        print("⚠️  Cannot select frame: frame is None")
        return None

    print(f"📸 Frame captured for annotation: {frame.size[0]}x{frame.size[1]}")
    return frame.copy()

def extract_coordinates_from_editor_data(editor_data):
    """Extract coordinates from Gradio ImageEditor data structure"""
    coordinates = []

    if editor_data is None:
        print("⚠️  Editor data is None")
        return coordinates

    try:
        print(f"🔍 Processing editor data type: {type(editor_data)}")

        # Handle different formats from Gradio ImageEditor
        img = None
        if isinstance(editor_data, dict):
            print(f"📋 Available keys: {list(editor_data.keys())}")

            # Priority order for finding the annotated image
            key_priority = ['composite', 'image', 'layers', 'background']
            for key in key_priority:
                if key in editor_data and editor_data[key] is not None:
                    img = editor_data[key]
                    print(f"🖼️  Using image from key: '{key}'")
                    break

            # Fallback: find any PIL Image
            if img is None:
                for key, value in editor_data.items():
                    if hasattr(value, 'save'):  # PIL Image check
                        img = value
                        print(f"🖼️  Found PIL image in key: '{key}'")
                        break
        else:
            # Direct PIL Image
            img = editor_data
            print("🖼️  Direct PIL image received")

        if img is not None:
            coordinates = analyze_image_for_annotations(img)
        else:
            print("❌ No valid image found for annotation")

    except Exception as e:
        print(f"❌ Error extracting coordinates: {e}")
        import traceback
        traceback.print_exc()

    print(f"📍 Final coordinate count: {len(coordinates)}")
    return coordinates

def analyze_image_for_annotations(image):
    """Analyze PIL image for drawn annotations using multiple methods"""
    coordinates = []

    try:
        print(f"🔍 Analyzing image: {image.size[0]}x{image.size[1]}")
        img_array = np.array(image)

        # Method 1: Edge detection for drawn lines
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array

        # Use multiple edge detection approaches
        edges = cv2.Canny(gray, 30, 100)  # Lower thresholds for better detection

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(f"🔍 Found {len(contours)} contours via edge detection")

        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area < 10:  # Lower threshold
                continue

            # Sample points from contour
            epsilon = 0.01 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            for point in approx:
                x, y = point[0]
                coordinates.append([int(x), int(y)])

        # Method 2: Color detection for common drawing colors
        color_coords = detect_annotation_colors(img_array)
        coordinates.extend(color_coords)

        # Method 3: Template matching for common shapes
        shape_coords = detect_drawn_shapes(gray)
        coordinates.extend(shape_coords)

        # Remove duplicates (points within 5 pixels of each other)
        if coordinates:
            coordinates = remove_duplicate_points(coordinates, min_distance=5)

    except Exception as e:
        print(f"❌ Error analyzing image: {e}")
        import traceback
        traceback.print_exc()

    return coordinates

def detect_annotation_colors(img_array):
    """Enhanced color detection for annotations"""
    coordinates = []

    # Expanded color palette for better detection
    colors_to_detect = [
        ([255, 0, 0], "red"),
        ([0, 255, 0], "green"),
        ([0, 0, 255], "blue"),
        ([255, 255, 0], "yellow"),
        ([255, 0, 255], "magenta"),
        ([0, 255, 255], "cyan"),
        ([0, 0, 0], "black"),
        ([255, 128, 0], "orange"),
        ([128, 0, 255], "purple"),
        ([255, 192, 203], "pink")
    ]

    for color, name in colors_to_detect:
        try:
            if len(img_array.shape) == 3:
                # Use HSV for better color detection
                hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)

                # Define range in HSV space for more robust detection
                lower = np.array([max(0, c-40) for c in color])
                upper = np.array([min(255, c+40) for c in color])
                mask = cv2.inRange(img_array, lower, upper)

                # Morphological operations to clean up mask
                kernel = np.ones((3,3), np.uint8)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            else:
                continue

            # Find coordinates
            y_coords, x_coords = np.where(mask > 0)

            if len(x_coords) > 20:  # Significant annotation
                print(f"🎨 Found {len(x_coords)} pixels of {name}")
                # Intelligent sampling - more points for smaller annotations
                sample_rate = max(3, len(x_coords) // 50)
                points = list(zip(x_coords[::sample_rate], y_coords[::sample_rate]))
                coordinates.extend([[int(x), int(y)] for x, y in points])

        except Exception as e:
            print(f"❌ Error detecting {name}: {e}")
            continue

    return coordinates

def detect_drawn_shapes(gray_image):
    """Detect common drawn shapes like circles, rectangles"""
    coordinates = []

    try:
        # Detect circles
        circles = cv2.HoughCircles(gray_image, cv2.HOUGH_GRADIENT, 1, 20,
                                   param1=50, param2=30, minRadius=10, maxRadius=100)

        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                # Add circle perimeter points
                for angle in range(0, 360, 30):  # Every 30 degrees
                    px = int(x + r * np.cos(np.radians(angle)))
                    py = int(y + r * np.sin(np.radians(angle)))
                    coordinates.append([px, py])
                print(f"🔵 Detected circle at ({x}, {y}) radius {r}")

        # Detect lines
        lines = cv2.HoughLinesP(gray_image, 1, np.pi/180, threshold=50,
                                minLineLength=20, maxLineGap=10)

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Sample points along the line
                num_points = max(2, int(np.sqrt((x2-x1)**2 + (y2-y1)**2) // 10))
                for i in range(num_points):
                    t = i / (num_points - 1) if num_points > 1 else 0
                    px = int(x1 + t * (x2 - x1))
                    py = int(y1 + t * (y2 - y1))
                    coordinates.append([px, py])
                print(f"📏 Detected line from ({x1},{y1}) to ({x2},{y2})")

    except Exception as e:
        print(f"❌ Shape detection error: {e}")

    return coordinates

def remove_duplicate_points(coordinates, min_distance=5):
    """Remove points that are too close to each other"""
    if not coordinates:
        return coordinates

    filtered = [coordinates[0]]  # Always keep first point

    for point in coordinates[1:]:
        too_close = False
        for existing in filtered:
            distance = np.sqrt((point[0] - existing[0])**2 + (point[1] - existing[1])**2)
            if distance < min_distance:
                too_close = True
                break
        if not too_close:
            filtered.append(point)

    print(f"🧹 Filtered {len(coordinates)} → {len(filtered)} points (removed duplicates)")
    return filtered

def send_annotation(editor_data):
    """Send annotation to server with enhanced error handling"""
    try:
        print("=" * 50)
        print("📤 SENDING ANNOTATION TO SERVER")
        print("=" * 50)

        if editor_data is None:
            return "❌ No annotation data to send"

        # Extract coordinates
        coordinates = extract_coordinates_from_editor_data(editor_data)

        if not coordinates:
            return "⚠️  No annotations detected. Try drawing something first!"

        # Get the annotated image
        img = None
        if isinstance(editor_data, dict):
            for key in ['composite', 'image', 'background']:
                if key in editor_data and editor_data[key] is not None:
                    img = editor_data[key]
                    break
        else:
            img = editor_data

        if img is None:
            return "❌ No image found in annotation data"

        # Convert to base64
        buf = BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()

        # Prepare payload
        payload = {
            "image": b64,
            "coordinates": coordinates,
            "annotation_count": len(coordinates),
            "timestamp": time.time()
        }

        print(f"📤 Sending {len(coordinates)} points to {SERVER_URL}/submit-annotation")

        # Send to server
        resp = requests.post(f"{SERVER_URL}/submit-annotation",
                           json=payload,
                           verify=VERIFY_SSL,
                           timeout=TIMEOUT)
        resp.raise_for_status()

        result = resp.json()
        success_msg = f"✅ SUCCESS! Sent {len(coordinates)} annotation points.\n📊 Server processed: {result.get('message', 'OK')}"
        print(success_msg)
        return success_msg

    except Exception as e:
        error_msg = f"❌ Send failed: {str(e)}"
        print(f"❌ Send error details: {e}")
        import traceback
        traceback.print_exc()
        return error_msg

def update_stream_display():
    """Update function for live stream display"""
    frame = get_current_frame()
    status = get_stream_status()
    return frame, status

def test_server_connection():
    """Test and report server connection status"""
    try:
        # Test basic connectivity
        health_resp = requests.get(f"{SERVER_URL}/health", verify=VERIFY_SSL, timeout=5)
        health_data = health_resp.json()

        # Test frame endpoint
        frame_resp = requests.get(f"{SERVER_URL}/latest-frame", verify=VERIFY_SSL, timeout=5)
        frame_data = frame_resp.json()

        status_msg = f"""
🏥 SERVER STATUS:
• Health: {health_data.get('status', 'unknown')}
• Frames available: {health_data.get('frames_available', 0)}
• Total frames received: {health_data.get('total_frames_received', 0)}
• Active connections: {health_data.get('active_connections', 0)}

📡 FRAME ENDPOINT:
• Status: {frame_data.get('status', 'unknown')}
• Has image: {'yes' if frame_data.get('image') else 'no'}
• Frame size: {frame_data.get('frame_size', 0)} bytes
"""

        print(status_msg)
        return True, status_msg

    except Exception as e:
        error_msg = f"❌ Server connection failed: {e}"
        print(error_msg)
        return False, error_msg

# Test server before starting UI
print("🧪 Testing server connection...")
connection_ok, connection_status = test_server_connection()

# Start streaming thread
if connection_ok:
    start_streaming()
    print("🚀 Background streaming started")
else:
    print("⚠️  Server connection issues detected")

# Enhanced UI
with gr.Blocks(css="""
    .gr-block { background: #111; color: #eee; }
    .status-indicator { font-family: monospace; background: #222; padding: 8px; border-radius: 4px; }
    .control-btn { width: auto !important; padding: 8px 16px !important; margin: 4px !important; }
""", title="XaiR Live Stream & Annotation") as demo:

    gr.Markdown("# 🔴 XaiR Live Preview & Annotation System")
    gr.Markdown("### Real-time camera feed with click-to-annotate functionality")

    # Status display
    with gr.Row():
        stream_status = gr.Textbox(label="📡 Stream Status",
                                  value=get_stream_status(),
                                  interactive=False,
                                  elem_classes=["status-indicator"])

    with gr.Row():
        # Control buttons
        start_btn = gr.Button("🟢 Start Stream", elem_classes=["control-btn"])
        stop_btn = gr.Button("🔴 Stop Stream", elem_classes=["control-btn"])
        test_btn = gr.Button("🧪 Test Connection", elem_classes=["control-btn"])

    with gr.Row():
        # Live stream column
        with gr.Column():
            gr.Markdown("### 📹 Live Camera Feed")
            live_display = gr.Image(type="pil",
                                   interactive=False,
                                   show_label=False,
                                   height=480,
                                   width=640)
            capture_btn = gr.Button("📸 Capture for Annotation", elem_classes=["control-btn"])

        # Annotation column
        with gr.Column():
            gr.Markdown("### ✏️ Annotation Editor")
            annotator = gr.ImageEditor(
                type="pil",
                interactive=True,
                show_label=False,
                height=480,
                width=640,
                brush=gr.Brush(
                    default_size=3,
                    colors=["#ff0000", "#00ff00", "#0000ff", "#ffff00", "#ff00ff", "#00ffff"]
                )
            )

    # Send controls
    with gr.Row():
        send_btn = gr.Button("📤 Extract & Send Annotations", elem_classes=["control-btn"])
        clear_btn = gr.Button("🗑️ Clear Annotations", elem_classes=["control-btn"])

    result_txt = gr.Textbox(label="📊 Operation Status",
                           interactive=False,
                           elem_classes=["status-indicator"])

    # Event handlers
    def refresh_display():
        return update_stream_display()

    def clear_annotator():
        return None

    def test_connection():
        ok, status = test_server_connection()
        return status

    # Set up event handlers
    start_btn.click(fn=start_streaming, outputs=result_txt)
    stop_btn.click(fn=stop_streaming, outputs=result_txt)
    test_btn.click(fn=test_connection, outputs=result_txt)

    capture_btn.click(fn=select_frame, inputs=live_display, outputs=annotator)
    send_btn.click(fn=send_annotation, inputs=annotator, outputs=result_txt)
    clear_btn.click(fn=clear_annotator, outputs=annotator)

    # Auto-refresh live display every 100ms
    demo.load(fn=refresh_display, outputs=[live_display, stream_status])

if __name__ == "__main__":
    print("🚀 Launching Gradio interface...")

    # Launch with optimized settings for live streaming
    demo.queue(max_size=50, default_concurrency_limit=10)
    demo.launch(
        server_name=args.ip,
        server_port=args.port,
        share=False,
        show_error=True,
        quiet=False,
        debug=True
    )
