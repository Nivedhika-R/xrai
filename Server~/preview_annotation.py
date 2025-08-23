
# # preview_annotation_with_working_audio.py - Fixed audio streaming

# import argparse
# from io import BytesIO
# import base64
# import requests
# import numpy as np
# import cv2
# from PIL import Image, ImageDraw
# import gradio as gr
# import json
# import time
# import threading

# parser = argparse.ArgumentParser(description="XaiR Live Preview & Annotate with Fixed Audio")
# parser.add_argument('--ip',      default='0.0.0.0', help='Host for Gradio server')
# parser.add_argument('--port',   type=int, default=7860,   help='Port for Gradio server')
# parser.add_argument('--backend', default='127.0.0.1',     help='Host/IP of your frame server')
# parser.add_argument('--bport',  type=int, default=8000,   help='Port of your frame server')
# parser.add_argument('--https', action='store_true',       help='Use HTTPS for backend')
# args = parser.parse_args()

# PROTO      = 'https' if args.https else 'http'
# SERVER_URL = f"{PROTO}://{args.backend}:{args.bport}"
# VERIFY_SSL = False
# TIMEOUT    = 5

# # Global variables for streaming
# latest_frame = None
# latest_pose_matrix = None
# streaming_active = False
# audio_streaming_active = False

# def fetch_latest_frame():
#     """Fetch a single frame from the server."""
#     global latest_pose_matrix
#     try:
#         resp = requests.get(f"{SERVER_URL}/latest-frame",
#                             timeout=TIMEOUT,
#                             verify=VERIFY_SSL)
#         resp.raise_for_status()
#         data = resp.json()
#         frame_base64 = data.get("image", "")
#         latest_pose_matrix = data.get("pose_matrix", None)

#         if not frame_base64:
#             return None

#         raw = base64.b64decode(frame_base64)
#         arr = np.frombuffer(raw, np.uint8)
#         cv_img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
#         rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
#         pil = Image.fromarray(rgb)
#         pil = pil.rotate(90, expand=True)
#         return pil
#     except Exception as e:
#         print("Fetch error:", e)
#         return None

# def continuous_frame_fetcher():
#     """Background thread to continuously fetch frames."""
#     global latest_frame, streaming_active
#     while streaming_active:
#         frame = fetch_latest_frame()
#         if frame is not None:
#             latest_frame = frame
#         time.sleep(0.1)

# def get_current_frame():
#     """Get the current frame for Gradio."""
#     global latest_frame
#     if latest_frame is not None:
#         return latest_frame
#     else:
#         return fetch_latest_frame()

# def start_streaming():
#     """Start the background streaming thread."""
#     global streaming_active
#     if streaming_active:
#         return "Already streaming"

#     streaming_active = True
#     fetch_thread = threading.Thread(target=continuous_frame_fetcher, daemon=True)
#     fetch_thread.start()
#     return "Live streaming started"

# def stop_streaming():
#     """Stop the streaming."""
#     global streaming_active
#     streaming_active = False
#     return "Streaming stopped"

# def select_frame(frame: Image.Image):
#     """Copy live frame into annotator."""
#     if frame is None:
#         return None
#     return {
#         "background": frame,
#         "layers": [],
#         "composite": None
#     }

# def extract_coordinates_from_layers(layers):
#     """Extract coordinates from the drawing layers in Gradio ImageEditor"""
#     coordinates = []

#     if not layers:
#         print("No layers found in ImageEditor data")
#         return coordinates

#     print(f"Found {len(layers)} layers to analyze")

#     for i, layer in enumerate(layers):
#         if layer is None:
#             continue

#         print(f"Processing layer {i}: {type(layer)}")

#         if hasattr(layer, 'convert'):
#             layer_rgba = layer.convert('RGBA')
#             layer_array = np.array(layer_rgba)
#         else:
#             layer_array = np.array(layer)

#         print(f"Layer {i} shape: {layer_array.shape}")

#         if layer_array.shape[2] == 4:
#             alpha_channel = layer_array[:, :, 3]
#             non_transparent = alpha_channel > 0
#             y_coords, x_coords = np.where(non_transparent)
#         else:
#             gray = np.mean(layer_array, axis=2)
#             non_black = gray > 10
#             y_coords, x_coords = np.where(non_black)

#         print(f"Layer {i}: Found {len(x_coords)} annotation pixels")

#         if len(x_coords) > 5:
#             points = list(zip(x_coords, y_coords))
#             sampled_points = points[::5]
#             layer_coords = [[int(x), int(y)] for x, y in sampled_points]
#             coordinates.extend(layer_coords)
#             print(f"Added {len(layer_coords)} points from layer {i}")

#     print(f"Total coordinates extracted: {len(coordinates)}")
#     return coordinates

# def create_visual_annotation_image(background, layers, coordinates):
#     """Create an image with visible red circles/lines marking the annotation points"""
#     if background is None:
#         return None

#     if isinstance(background, np.ndarray):
#         background = Image.fromarray(background)
#     elif not hasattr(background, 'copy'):
#         return None

#     annotated_img = background.copy()
#     draw = ImageDraw.Draw(annotated_img)

#     if coordinates:
#         for coord in coordinates:
#             x, y = coord[0], coord[1]
#             radius = 4
#             draw.ellipse([x-radius, y-radius, x+radius, y+radius],
#                         fill='red', outline='darkred', width=2)

#         if len(coordinates) > 1:
#             for i in range(len(coordinates)-1):
#                 x1, y1 = coordinates[i]
#                 x2, y2 = coordinates[i+1]
#                 distance = ((x2-x1)**2 + (y2-y1)**2)**0.5
#                 if distance < 50:
#                     draw.line([x1, y1, x2, y2], fill='red', width=2)

#     return annotated_img

# def send_annotation(editor_data):
#     """POST annotated image and coordinates back to your server."""
#     global latest_pose_matrix

#     try:
#         print("=" * 50)
#         print("🎯 PROCESSING ANNOTATION")
#         print("=" * 50)
#         print(f"Editor data type: {type(editor_data)}")

#         if editor_data is None:
#             return "❌ No data received from ImageEditor"

#         if isinstance(editor_data, dict):
#             background = editor_data.get("background")
#             layers = editor_data.get("layers", [])
#             composite = editor_data.get("composite")

#             print(f"📁 Background: {type(background)}")
#             print(f"🎨 Layers: {len(layers) if layers else 0}")
#             print(f"🖼️ Composite: {type(composite)}")

#             coordinates = extract_coordinates_from_layers(layers)
#         else:
#             return f"❌ Unexpected data format: {type(editor_data)}"

#         if not coordinates:
#             return "❌ No annotations found! Please draw something on the image using the brush tools."

#         image_to_send = composite if composite is not None else background

#         if image_to_send is None:
#             return "❌ No image data available"

#         if isinstance(image_to_send, np.ndarray):
#             image_to_send = Image.fromarray(image_to_send)

#         visual_annotation_img = create_visual_annotation_image(background, layers, coordinates)

#         timestamp = int(time.time())
#         visual_filename = f"annotated_visual_{timestamp}.png"
#         if visual_annotation_img:
#             visual_annotation_img.save(visual_filename)
#             print(f"💾 Saved visual annotation: {visual_filename}")

#         buf = BytesIO()
#         if isinstance(image_to_send, np.ndarray):
#             pil_image = Image.fromarray(image_to_send)
#             pil_image.save(buf, format="PNG")
#         else:
#             image_to_send.save(buf, format="PNG")
#         b64 = base64.b64encode(buf.getvalue()).decode()

#         payload = {
#             "image": b64,
#             "coordinates": coordinates,
#             "annotation_count": len(coordinates),
#             "pose_matrix": latest_pose_matrix if latest_pose_matrix else None,
#             "timestamp": timestamp
#         }

#         print(f"📤 Sending {len(coordinates)} coordinates to server...")

#         resp = requests.post(f"{SERVER_URL}/submit-annotation",
#                              json=payload,
#                              verify=VERIFY_SSL,
#                              timeout=TIMEOUT)
#         resp.raise_for_status()

#         result = resp.json()

#         success_msg = f"✅ Successfully sent {len(coordinates)} annotation points!\n"
#         success_msg += f"📁 Visual annotation saved: {visual_filename}\n"
#         if latest_pose_matrix:
#             success_msg += f"📐 Pose matrix included\n"
#         success_msg += f"🖥️ Server response: {result.get('message', 'OK')}"

#         print("✅ Annotation sent successfully!")
#         return success_msg

#     except Exception as e:
#         print(f"❌ Error processing annotation: {e}")
#         import traceback
#         traceback.print_exc()
#         return f"❌ Error: {e}"

# def auto_update_display():
#     """Return the latest frame if streaming is active."""
#     global streaming_active, latest_frame
#     if streaming_active and latest_frame is not None:
#         return latest_frame
#     return gr.update()

# # ============ SIMPLIFIED AUDIO FUNCTIONS ============

# def send_audio_via_http(audio_data):
#     """Send audio data via HTTP POST (simple approach)"""
#     global audio_streaming_active
    
#     if not audio_streaming_active:
#         return "🔇 Audio streaming not active"
    
#     try:
#         if audio_data is None:
#             return "❌ No audio data received"
        
#         # Handle Gradio audio data format
#         if isinstance(audio_data, tuple):
#             sample_rate, audio_array = audio_data
#         else:
#             # Assume it's just the audio array with default sample rate
#             sample_rate = 16000
#             audio_array = audio_data
        
#         # Convert to the right format
#         if isinstance(audio_array, np.ndarray):
#             # Convert to int16 range for transmission
#             if audio_array.dtype == np.float32 or audio_array.dtype == np.float64:
#                 # Convert from float [-1,1] to int16 [-32768, 32767]
#                 audio_int16 = (audio_array * 32767).astype(np.int16)
#             else:
#                 audio_int16 = audio_array.astype(np.int16)
            
#             # Convert to list for JSON serialization
#             audio_list = audio_int16.tolist()
#         else:
#             audio_list = list(audio_array)
        
#         # Prepare payload
#         payload = {
#             "audioData": audio_list,
#             "sampleRate": int(sample_rate),
#             "timestamp": int(time.time() * 1000)
#         }
        
#         print(f"📤 Sending {len(audio_list)} audio samples to server...")
        
#         # Send to server
#         resp = requests.post(f"{SERVER_URL}/unity-send-audio",
#                              json=payload,
#                              verify=VERIFY_SSL,
#                              timeout=TIMEOUT)
#         resp.raise_for_status()
        
#         result = resp.json()
#         print(f"✅ Audio sent successfully: {result}")
        
#         return f"✅ Sent {len(audio_list)} audio samples to Unity!"
        
#     except Exception as e:
#         print(f"❌ Error sending audio: {e}")
#         import traceback
#         traceback.print_exc()
#         return f"❌ Audio error: {e}"

# def start_live_audio():
#     """Start live audio streaming"""
#     global audio_streaming_active
    
#     # Test server connection first
#     try:
#         test_resp = requests.get(f"{SERVER_URL}/latest-frame", timeout=2, verify=VERIFY_SSL)
#         if test_resp.status_code != 200:
#             return f"❌ Server not reachable at {SERVER_URL}"
#     except Exception as e:
#         return f"❌ Cannot connect to server: {e}"
    
#     audio_streaming_active = True
    
#     return f"🎤 Audio streaming enabled! Server: {SERVER_URL}\n" + \
#            "📝 Record audio using the microphone component below, then click 'Send Audio'."

# def stop_live_audio():
#     """Stop live audio streaming"""
#     global audio_streaming_active
#     audio_streaming_active = False
#     return "🔇 Audio streaming disabled"

# def get_audio_status():
#     """Get current audio streaming status"""
#     global audio_streaming_active
    
#     if audio_streaming_active:
#         return f"🎤 READY - Audio will be sent to {SERVER_URL}/unity-send-audio"
#     else:
#         return "🔇 Audio streaming disabled - Click 'START LIVE AUDIO' to enable"

# # ============ UI SETUP ============

# with gr.Blocks(css="""
#     .gr-block { background: #111; color: #eee; }
#     .gr-button { margin-top: 8px; }
#     #send-annotation-btn button { width:auto; padding: 6px 12px; font-size: 14px; max-width: 120px; }
#     #capture-btn button { width:auto; padding: 6px 12px; font-size: 14px; max-width: 100px; }
#     .audio-section { border: 2px solid #007acc; padding: 15px; border-radius: 8px; margin: 10px 0; background: #222; }
#     .live-indicator { color: #00ff00; font-weight: bold; }
# """) as demo:

#     gr.Markdown("## 🎤 XaiR Live Preview & Annotate with Audio")

#     with gr.Row():
#         # ─ Live stream column ─
#         with gr.Column():
#             gr.Markdown("### 📹 Live Video Feed")
#             live_display = gr.Image(type="pil",
#                                     interactive=False,
#                                     show_label=False,
#                                     height=480,
#                                     width=640)

#             with gr.Row():
#                 start_btn = gr.Button("🎬 Start Stream")
#                 stop_btn = gr.Button("⏹️ Stop Stream")

#             with gr.Row():
#                 refresh_btn = gr.Button("🔄 Refresh Feed")
#                 capture_btn = gr.Button("📸 Capture Frame", elem_id="capture-btn")

#             stream_status = gr.Textbox(label="Stream Status", interactive=False)

#         # ─ Annotation column ─
#         with gr.Column():
#             gr.Markdown("### 🖌️ Draw / Click to Annotate")
#             gr.Markdown("**Instructions:**")
#             gr.Markdown("1. Click 'Capture Frame' to get current image")
#             gr.Markdown("2. Use brush tools (🖌️) to draw annotations")
#             gr.Markdown("3. Click 'Extract & Send Coordinates' to send data")

#             annotator = gr.ImageEditor(
#                 interactive=True,
#                 show_label=False,
#                 height=480,
#                 width=640,
#                 brush=gr.Brush(
#                     default_size=5,
#                     colors=["#ff0000", "#00ff00", "#0000ff", "#ffff00"],
#                     default_color="#ff0000"
#                 ),
#                 layers=True
#             )

#     # ============ SIMPLIFIED AUDIO SECTION ============
#     with gr.Group(elem_classes=["audio-section"]):
#         gr.Markdown("### 🎤 **AUDIO COMMUNICATION**")
#         gr.Markdown("Send voice messages to Unity")
        
#         with gr.Row():
#             with gr.Column():
#                 # Audio recording component
#                 audio_input = gr.Audio(
#                     label="🎤 Record Your Voice Message",
#                     type="numpy",
#                     sources=["microphone"],
#                     interactive=True
#                 )
                
#                 with gr.Row():
#                     start_audio_btn = gr.Button("🎤 START AUDIO STREAMING", variant="primary")
#                     stop_audio_btn = gr.Button("🔇 STOP AUDIO", variant="secondary")
                
#                 send_audio_btn = gr.Button("📤 Send Recorded Audio to Unity", variant="primary")
                
#             with gr.Column():
#                 audio_status_display = gr.Textbox(
#                     label="🎤 Audio Status", 
#                     interactive=False,
#                     value="Click 'START AUDIO STREAMING' to enable audio communication"
#                 )
                
#                 audio_result = gr.Textbox(
#                     label="📤 Send Results", 
#                     interactive=False,
#                     lines=3
#                 )
        
#         gr.Markdown("**📋 How to use audio:**")
#         gr.Markdown("1. Click '🎤 START AUDIO STREAMING' to enable")
#         gr.Markdown("2. Use the microphone to record your message")
#         gr.Markdown("3. Click '📤 Send Recorded Audio to Unity'")
#         gr.Markdown("4. Unity will receive your audio via the server")

#     # Main action buttons
#     send_btn = gr.Button("🚀 Extract & Send Coordinates", elem_id="send-annotation-btn")
#     result_txt = gr.Textbox(label="📊 Send Status", interactive=False, lines=6)

#     # ============ EVENT HANDLERS ============
    
#     # Video streaming events
#     start_btn.click(fn=start_streaming, outputs=stream_status)
#     stop_btn.click(fn=stop_streaming, outputs=stream_status)
#     refresh_btn.click(fn=get_current_frame, outputs=live_display)
#     capture_btn.click(fn=select_frame, inputs=live_display, outputs=annotator)
#     send_btn.click(fn=send_annotation, inputs=annotator, outputs=result_txt)
    
#     # Audio streaming events
#     start_audio_btn.click(fn=start_live_audio, outputs=audio_status_display)
#     stop_audio_btn.click(fn=stop_live_audio, outputs=audio_status_display)
#     send_audio_btn.click(fn=send_audio_via_http, inputs=audio_input, outputs=audio_result)

#     # Auto-refresh timers
#     refresh_timer = gr.Timer(0.1)  # Update video every 100ms
#     refresh_timer.tick(fn=auto_update_display, outputs=live_display)
    
#     # Audio status update timer
#     audio_timer = gr.Timer(2.0)  # Update audio status every 2 seconds
#     audio_timer.tick(fn=get_audio_status, outputs=audio_status_display)

# # Launch the Gradio app
# demo.queue()
# demo.launch(server_name=args.ip,
#             server_port=args.port,
#             share=False)

# Run : python preview_annotation.py --backend <ip address> --bport 8000 --https

import argparse
from io import BytesIO
import base64
import requests
import numpy as np
import cv2
from PIL import Image, ImageDraw
import gradio as gr
import json
import time
import threading

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

# Global variables for streaming
latest_frame = None
latest_pose_matrix = None  # Store the pose matrix
streaming_active = False
ui_update_thread = None
live_display_component = None

def fetch_latest_frame():
    """Fetch a single frame from the server."""
    global latest_pose_matrix
    try:
        resp = requests.get(f"{SERVER_URL}/latest-frame",
                            timeout=TIMEOUT,
                            verify=VERIFY_SSL)
        resp.raise_for_status()
        data = resp.json()
        frame_base64 = data.get("image", "")

        # Extract pose matrix if available
        latest_pose_matrix = data.get("pose_matrix", None)

        if not frame_base64:
            return None

        raw = base64.b64decode(frame_base64)
        arr = np.frombuffer(raw, np.uint8)
        cv_img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)

        # Convert to PIL and rotate to correct orientation
        pil = Image.fromarray(rgb)
        pil = pil.rotate(90, expand=True)  # Adjust rotation as needed
        return pil
    except Exception as e:
        print("Fetch error:", e)
        return None

def continuous_frame_fetcher():
    """Background thread to continuously fetch frames."""
    global latest_frame, streaming_active
    while streaming_active:
        frame = fetch_latest_frame()
        if frame is not None:
            latest_frame = frame
        time.sleep(0.1)  # Fetch at ~10 FPS

def ui_updater():
    """Thread to update the Gradio UI with latest frames."""
    global streaming_active, latest_frame, live_display_component
    while streaming_active:
        if latest_frame is not None and live_display_component is not None:
            try:
                # Update the live display component
                live_display_component.update(value=latest_frame)
            except:
                pass
        time.sleep(0.1)  # Update UI at ~10 FPS

def get_current_frame():
    """Get the current frame for Gradio."""
    global latest_frame
    if latest_frame is not None:
        return latest_frame
    else:
        # If no cached frame, fetch one immediately
        return fetch_latest_frame()

def start_streaming():
    """Start the background streaming thread."""
    global streaming_active, ui_update_thread
    if streaming_active:
        return "Already streaming"

    streaming_active = True

    # Start frame fetching thread
    fetch_thread = threading.Thread(target=continuous_frame_fetcher, daemon=True)
    fetch_thread.start()

    return "Live streaming started"

def stop_streaming():
    """Stop the streaming."""
    global streaming_active, ui_update_thread
    streaming_active = False
    if ui_update_thread and ui_update_thread.is_alive():
        ui_update_thread.join(timeout=1)
    return "Streaming stopped"

def select_frame(frame: Image.Image):
    """Copy live frame into annotator."""
    if frame is None:
        return None
    # Return the frame in the correct ImageEditor format
    return {
        "background": frame,
        "layers": [],
        "composite": None
    }

def extract_coordinates_from_layers(layers):
    """
    Extract coordinates from the drawing layers in Gradio ImageEditor
    """
    coordinates = []

    if not layers:
        print("No layers found in ImageEditor data")
        return coordinates

    print(f"Found {len(layers)} layers to analyze")

    for i, layer in enumerate(layers):
        if layer is None:
            continue

        print(f"Processing layer {i}: {type(layer)}")

        # Convert layer to numpy array
        if hasattr(layer, 'convert'):  # PIL Image
            # Convert to RGBA to handle transparency
            layer_rgba = layer.convert('RGBA')
            layer_array = np.array(layer_rgba)
        else:
            layer_array = np.array(layer)

        print(f"Layer {i} shape: {layer_array.shape}")

        # Find non-transparent pixels (alpha > 0 for RGBA, or any non-zero for RGB)
        if layer_array.shape[2] == 4:  # RGBA
            # Find pixels where alpha > 0 (not transparent)
            alpha_channel = layer_array[:, :, 3]
            non_transparent = alpha_channel > 0
            y_coords, x_coords = np.where(non_transparent)
        else:  # RGB
            # Find pixels that are not black (assuming black is background)
            gray = np.mean(layer_array, axis=2)
            non_black = gray > 10  # Threshold for non-black pixels
            y_coords, x_coords = np.where(non_black)

        print(f"Layer {i}: Found {len(x_coords)} annotation pixels")

        if len(x_coords) > 5:  # Only process if we found enough pixels
            # Sample points to avoid too many coordinates
            points = list(zip(x_coords, y_coords))
            # Sample every 5th point for drawn annotations
            sampled_points = points[::5]
            layer_coords = [[int(x), int(y)] for x, y in sampled_points]
            coordinates.extend(layer_coords)
            print(f"Added {len(layer_coords)} points from layer {i}")

    print(f"Total coordinates extracted: {len(coordinates)}")
    return coordinates

def create_visual_annotation_image(background, layers, coordinates):
    """
    Create an image with visible red circles/lines marking the annotation points
    """
    if background is None:
        return None

    # Convert background to PIL Image if it's numpy array
    if isinstance(background, np.ndarray):
        background = Image.fromarray(background)
    elif not hasattr(background, 'copy'):
        return None

    # Start with the background
    annotated_img = background.copy()
    draw = ImageDraw.Draw(annotated_img)

    # Draw red circles at each coordinate point
    if coordinates:
        for coord in coordinates:
            x, y = coord[0], coord[1]
            radius = 4
            # Draw red circle with border
            draw.ellipse([x-radius, y-radius, x+radius, y+radius],
                        fill='red', outline='darkred', width=2)

        # Connect nearby points with lines
        if len(coordinates) > 1:
            for i in range(len(coordinates)-1):
                x1, y1 = coordinates[i]
                x2, y2 = coordinates[i+1]
                distance = ((x2-x1)**2 + (y2-y1)**2)**0.5
                # Only connect points that are close together (part of same stroke)
                if distance < 50:
                    draw.line([x1, y1, x2, y2], fill='red', width=2)

    return annotated_img

def send_annotation(editor_data):
    """POST annotated image and coordinates back to your server."""
    global latest_pose_matrix

    try:
        print("=" * 50)
        print("🎯 PROCESSING ANNOTATION")
        print("=" * 50)
        print(f"Editor data type: {type(editor_data)}")

        if editor_data is None:
            return "❌ No data received from ImageEditor"

        # Handle the ImageEditor data format: {"background", "layers", "composite"}
        if isinstance(editor_data, dict):
            background = editor_data.get("background")
            layers = editor_data.get("layers", [])
            composite = editor_data.get("composite")

            print(f"📁 Background: {type(background)}")
            print(f"🎨 Layers: {len(layers) if layers else 0}")
            print(f"🖼️ Composite: {type(composite)}")

            # Extract coordinates from the drawing layers
            coordinates = extract_coordinates_from_layers(layers)

        else:
            return f"❌ Unexpected data format: {type(editor_data)}"

        # Check if any annotations were found
        if not coordinates:
            return "❌ No annotations found! Please draw something on the image using the brush tools."

        # Use composite image if available, otherwise background
        image_to_send = composite if composite is not None else background

        if image_to_send is None:
            return "❌ No image data available"

        # Convert to PIL Image if it's numpy array
        if isinstance(image_to_send, np.ndarray):
            image_to_send = Image.fromarray(image_to_send)

        # Create visual annotation image for saving
        visual_annotation_img = create_visual_annotation_image(background, layers, coordinates)

        # Save the visual annotation image
        timestamp = int(time.time())
        visual_filename = f"annotated_visual_{timestamp}.png"
        if visual_annotation_img:
            visual_annotation_img.save(visual_filename)
            print(f"💾 Saved visual annotation: {visual_filename}")

        # Also save the composite/layers for debugging
        debug_filename = f"debug_composite_{timestamp}.png"
        if composite is not None:
            try:
                if isinstance(composite, np.ndarray):
                    composite_pil = Image.fromarray(composite)
                    composite_pil.save(debug_filename)
                else:
                    composite.save(debug_filename)
                print(f"🔍 Saved debug composite: {debug_filename}")
            except Exception as e:
                print(f"⚠️ Could not save debug composite: {e}")
                debug_filename = None

        # Convert image to base64 for sending to server
        buf = BytesIO()
        if isinstance(image_to_send, np.ndarray):
            # Convert numpy array to PIL Image first
            pil_image = Image.fromarray(image_to_send)
            pil_image.save(buf, format="PNG")
        else:
            # Already a PIL Image
            image_to_send.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()

        # Prepare payload with coordinates and pose matrix
        payload = {
            "image": b64,
            "coordinates": coordinates,
            "annotation_count": len(coordinates),
            "pose_matrix": latest_pose_matrix if latest_pose_matrix else None,
            "timestamp": timestamp
        }

        print(f"📤 Sending {len(coordinates)} coordinates to server...")

        resp = requests.post(f"{SERVER_URL}/submit-annotation",
                             json=payload,
                             verify=VERIFY_SSL,
                             timeout=TIMEOUT)
        resp.raise_for_status()

        result = resp.json()

        # Create success message
        success_msg = f"✅ Successfully sent {len(coordinates)} annotation points!\n"
        success_msg += f"📁 Visual annotation saved: {visual_filename}\n"
        if debug_filename:
            success_msg += f"🔍 Debug image saved: {debug_filename}\n"
        if latest_pose_matrix:
            success_msg += f"📐 Pose matrix included\n"
        success_msg += f"🖥️ Server response: {result.get('message', 'OK')}"

        print("✅ Annotation sent successfully!")
        return success_msg

    except Exception as e:
        print(f"❌ Error processing annotation: {e}")
        import traceback
        traceback.print_exc()
        return f"❌ Error: {e}"

# Auto-update function for live streaming
def auto_update_display():
    """Return the latest frame if streaming is active."""
    global streaming_active, latest_frame
    if streaming_active and latest_frame is not None:
        return latest_frame
    return gr.update()

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

            with gr.Row():
                start_btn = gr.Button("Start Stream")
                stop_btn = gr.Button("Stop Stream")

            with gr.Row():
                refresh_btn = gr.Button("Refresh Feed")
                capture_btn = gr.Button("Capture Frame", elem_id="capture-btn")

            stream_status = gr.Textbox(label="Stream Status", interactive=False)

        # ─ Annotation column ─
        with gr.Column():
            gr.Markdown("### Draw / Click to Annotate")
            gr.Markdown("**Instructions:**")
            gr.Markdown("1. Click 'Capture Frame' to get current image")
            gr.Markdown("2. Use brush tools (🖌️) to draw annotations")
            gr.Markdown("3. Click 'Extract & Send Coordinates' to send data")

            annotator = gr.ImageEditor(
                interactive=True,
                show_label=False,
                height=480,
                width=640,
                brush=gr.Brush(
                    default_size=5,
                    colors=["#ff0000", "#00ff00", "#0000ff", "#ffff00"],
                    default_color="#ff0000"
                ),
                layers=True  # Enable layers for better drawing detection
            )

    send_btn = gr.Button("🚀 Extract & Send Coordinates", elem_id="send-annotation-btn")
    result_txt = gr.Textbox(label="📊 Send Status", interactive=False, lines=6)

    # Event handlers
    start_btn.click(fn=start_streaming, outputs=stream_status)
    stop_btn.click(fn=stop_streaming, outputs=stream_status)
    refresh_btn.click(fn=get_current_frame, outputs=live_display)
    capture_btn.click(fn=select_frame, inputs=live_display, outputs=annotator)
    send_btn.click(fn=send_annotation, inputs=annotator, outputs=result_txt)

    # Auto-refresh timer - this creates a continuous update loop
    refresh_timer = gr.Timer(0.1)  # Update every 100ms
    refresh_timer.tick(fn=auto_update_display, outputs=live_display)

# launch the Gradio app
demo.queue()
demo.launch(server_name=args.ip,
            server_port=args.port,
            share=False)
