# Modified XaiR Server with Annotation Support

import os
import ssl
import time
import json
import asyncio
import base64
import argparse
import logging

from threading import Thread
from queue import Queue
from collections import deque, namedtuple

import cv2
from PIL import Image, ImageEnhance
from io import BytesIO

import numpy as np

from collections import defaultdict
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCIceCandidate

from logger import logger
from constants import *
from frame import Frame
from board_tracker import BoardTracker
from debugger import Debugger

server_id = 0
next_client_id = 1
consume_tasks = {}
created_offers = {}
created_answers = {}
pcs = {}  # client_id: RTCPeerConnection
recorders = {}  # client_id: MediaRecorder
ices = defaultdict(list)

msg_queue = Queue()
image_bgr = deque()
latest_frame_data = None  # Store complete frame data including matrices

board_tracker = BoardTracker()
debugger = None

@web.middleware
async def cors_middleware(request, handler):
    if request.method == "OPTIONS":
        return web.Response(status=200)
    response = await handler(request)
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "*"
    return response

# Frame data structure
FrameData = namedtuple("FrameData", [
    "client_id", "image", "camera_matrix", "projection_matrix", 
    "distortion_matrix", "timestamp", "camera_to_world"
])

def detect_annotations_in_image(original_image, annotated_image):
    """Detect annotation points by comparing original and annotated images"""
    try:
        if isinstance(original_image, Image.Image):
            orig_array = np.array(original_image)
        else:
            orig_array = original_image
            
        if isinstance(annotated_image, Image.Image):
            annot_array = np.array(annotated_image)
        else:
            annot_array = annotated_image
        
        # Ensure same shape
        if orig_array.shape != annot_array.shape:
            logger.warning("Image shape mismatch")
            return []
        
        # Find differences
        diff = np.abs(orig_array.astype(np.float32) - annot_array.astype(np.float32))
        diff_gray = np.mean(diff, axis=2) if len(diff.shape) == 3 else diff
        
        # Threshold for significant changes
        threshold = 30
        mask = diff_gray > threshold
        
        # Find contours
        mask_uint8 = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        annotation_points = []
        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                area = cv2.contourArea(contour)
                
                if area > 50:  # Minimum area threshold
                    annotation_points.append({
                        'x': cx,
                        'y': cy,
                        'area': area
                    })
        
        return annotation_points
        
    except Exception as e:
        logger.error(f"Error detecting annotations: {e}")
        return []

# GET /latest-frame - FIXED VERSION
async def get_latest_frame(request):
    global image_bgr
    if len(image_bgr) == 0:
        return web.json_response({"image": None})

    frame = image_bgr[-1]
    # logger.debug("Sending latest frame to client")
    _, buffer = cv2.imencode('.jpg', frame)  # FIXED: use frame directly, not image_bgr
    frame_base64 = base64.b64encode(buffer).decode('utf-8')
    return web.json_response({"image": frame_base64})

# NEW: POST /submit-annotation - Handle annotations from Gradio
async def submit_annotation(request):
    global latest_frame_data, image_bgr, msg_queue
    
    try:
        data = await request.json()
        base64_image = data.get("image")
        client_id = data.get("client_id", "gradio")
        
        if not base64_image:
            return web.json_response({"error": "No image data"}, status=400)
        
        # Decode annotated image
        image_bytes = base64.b64decode(base64_image)
        annotated_image = Image.open(BytesIO(image_bytes))
        
        # Save annotation
        timestamp = int(time.time())
        annotation_filename = f"annotation_{timestamp}.png"
        annotated_image.save(annotation_filename)
        
        print("\n" + "="*60)
        print("🎯 ANNOTATION RECEIVED FROM GRADIO")
        print("="*60)
        print(f"📅 Timestamp: {timestamp}")
        print(f"💾 Saved as: {annotation_filename}")
        print(f"📐 Image Size: {annotated_image.size}")
        
        # Detect annotation points
        annotation_points = []
        if len(image_bgr) > 0:
            original_image = image_bgr[-1]
            original_pil = Image.fromarray(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
            
            detected_points = detect_annotations_in_image(original_pil, annotated_image)
            
            print(f"🔍 Detected {len(detected_points)} annotation points:")
            print("-" * 40)
            
            for i, point in enumerate(detected_points):
                pixel_x = point['x']
                pixel_y = point['y']
                area = point['area']
                
                annotation_points.append({
                    'pixel_x': pixel_x,
                    'pixel_y': pixel_y,
                    'area': area
                })
                
                print(f"📍 Point {i+1}:")
                print(f"   └── Coordinates: ({pixel_x}, {pixel_y})")
                print(f"   └── Area: {area} pixels")
                print()
        
        # Print camera data if available
        if latest_frame_data:
            print("📷 CAMERA DATA:")
            print("-" * 40)
            print(f"🔗 Client ID: {latest_frame_data.client_id}")
            print(f"⏰ Frame Timestamp: {latest_frame_data.timestamp}")
            
            if latest_frame_data.camera_to_world is not None:
                print("🌍 Camera-to-World Matrix:")
                print(latest_frame_data.camera_to_world)
                print()
            
            if latest_frame_data.camera_matrix is not None:
                print("📐 Camera Matrix (Intrinsics):")
                print(latest_frame_data.camera_matrix)
                print()
            
            if latest_frame_data.projection_matrix is not None:
                print("🎯 Projection Matrix:")
                print(latest_frame_data.projection_matrix)
                print()
            
            # Send annotation data to Unity for hit testing and 3D placement
            if annotation_points:
                unity_message = {
                    "clientID": latest_frame_data.client_id,
                    "type": "annotation_hittest",
                    "content": {
                        "annotation_points": annotation_points,
                        "timestamp": timestamp,
                        "camera_to_world": latest_frame_data.camera_to_world.tolist() if latest_frame_data.camera_to_world is not None else None,
                        "camera_matrix": latest_frame_data.camera_matrix.tolist() if latest_frame_data.camera_matrix is not None else None,
                        "projection_matrix": latest_frame_data.projection_matrix.tolist() if latest_frame_data.projection_matrix is not None else None
                    }
                }
                
                # Queue message to be sent to Unity via WebRTC
                msg_queue.put(unity_message)
                print("📤 Sent annotation data to Unity client for 3D placement")
        else:
            print("⚠️  No camera data available - annotations saved but cannot send to Unity")
        
        print("="*60)
        print()
        
        return web.json_response({
            "status": "success",
            "message": f"Annotation processed successfully",
            "points_count": len(annotation_points),
            "filename": annotation_filename
        })
        
    except Exception as e:
        logger.error(f"Error processing annotation: {e}")
        return web.json_response({"error": str(e)}, status=500)

# POST /login
async def login(request):
    global next_client_id
    client_id = next_client_id
    next_client_id += 1
    
    # Enhanced logging
    print(f"\n🔥 LOGIN ATTEMPT DETECTED!")
    print(f"📱 Client IP: {request.remote}")
    print(f"🆔 Assigned Client ID: {client_id}")
    print(f"📊 Headers: {dict(request.headers)}")
    print(f"🌐 Method: {request.method}")
    print("-" * 50)
    
    logger.info("User %s logged in from %s", client_id, request.remote)
    return web.Response(text=str(client_id))

# POST /logout/{id}
async def logout(request):
    client_id = int(request.match_info["id"])
    logger.info("User %s logged out", client_id)

    if client_id in consume_tasks:
        consume_tasks[client_id].cancel()
        del consume_tasks[client_id]

    if client_id in pcs:
        pc = pcs[client_id]
        for transceiver in pc.getTransceivers():
            track = transceiver.receiver.track
            if track:
                try:
                    track.stop()
                except Exception as e:
                    logger.warning(f"Failed to stop track for user {client_id}: {e}")
        await pc.close()
        del pcs[client_id]

    if client_id in recorders:
        await recorders[client_id].stop()
        del recorders[client_id]

    ices.pop(client_id, None)
    return web.Response(status=200)

async def dummy_consume(track):
    while True:
        await track.recv()

# POST /post_offer/{id}
async def post_offer(request):
    global msg_queue

    client_id = int(request.match_info["id"])
    params = await request.json()

    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    # Store reference to the offer (for other peers to discover)
    created_offers[client_id] = {
        "id": client_id,
        "has_audio": "audio" in params.get("sdp", ""),
        "has_video": "video" in params.get("sdp", ""),
    }

    # Create peer connection
    pc = RTCPeerConnection()
    pcs[client_id] = pc

    @pc.on("track")
    async def on_track(track):
        if track.kind == "video":
            logger.info("Receiving video from client (we dont send video so we should never get here...)")
        elif track.kind == "audio":
            logger.info("Receiving audio from client!")

    @pc.on("datachannel")
    def on_datachannel(channel):
        logger.info(f"DataChannel created: {channel.label}")
        msg = {
            "clientID": client_id,
            "type": "greeting",
            "content": "Hello from XaiR!"
        }
        payload = json.dumps(msg)
        channel.send(payload)

        async def listen_for_msgs():
            while True:
                msg = await asyncio.get_event_loop().run_in_executor(None, msg_queue.get)
                payload = json.dumps(msg)
                try:
                    channel.send(payload)
                except Exception as e:
                    logger.error(f"Failed to send message via DataChannel: {e}")
                    break

        asyncio.create_task(listen_for_msgs())

        @channel.on("message")
        def on_message(message):
            logger.info(f"Received message via DataChannel: {message}")

    logger.info("User %s posted SDP offer", client_id)

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    created_answers[client_id] = {
        "id": server_id,
        "answer": {
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type
        }
    }

    return web.json_response({
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type
    })

# POST /post_answer/{from_id}/{to_id}
async def post_answer(request):
    from_id = int(request.match_info["from_id"])
    to_id = int(request.match_info["to_id"])

    try:
        data = await request.json()
        logger.info("User %s posted answer to user %s", from_id, to_id)

        created_offers.pop(to_id, None)

        created_answers[to_id] = {
            "id": from_id,
            "answer": {
                "sdp": data["sdp"],
                "type": data["type"]
            }
        }

        return web.Response(status=200)

    except Exception as e:
        logger.warning("Failed to save answer from %s to %s: %s", from_id, to_id, e)
        return web.Response(status=400)

# POST /post_ice/{id}
async def post_ice(request):
    client_id = int(request.match_info["id"])
    if client_id not in pcs:
        return web.Response(status=404)

    try:
        data = await request.json()
        ip = data['candidate'].split(' ')[4]
        port = data['candidate'].split(' ')[5]
        protocol = data['candidate'].split(' ')[7]
        priority = data['candidate'].split(' ')[3]
        foundation = data['candidate'].split(' ')[0]
        component = data['candidate'].split(' ')[1]
        type = data['candidate'].split(' ')[7]

        rtc_candidate = RTCIceCandidate(
            ip=ip,
            port=port,
            protocol=protocol,
            priority=priority,
            foundation=foundation,
            component=component,
            type=type,
            sdpMid=data["sdpMid"],
            sdpMLineIndex=int(data["sdpMLineIndex"])
        )
        await pcs[client_id].addIceCandidate(rtc_candidate)
        logger.info("Added ICE candidate for user %s", client_id)
    except Exception as e:
        logger.warning("Failed to add ICE for %s: %s", client_id, e)

    return web.Response(status=200)

# POST /consume_ices/{id}
async def consume_ices(request):
    client_id = int(request.match_info["id"])
    ice_list = ices[client_id]
    ices[client_id] = []
    if ice_list:
        logger.info("Someone consumed ICEs from user %s", client_id)
    return web.json_response({"ices": ice_list})

async def get_offers(request):
    return web.json_response(created_offers)

async def get_answers(request):
    return web.json_response(created_answers)

Frame = namedtuple("Frame", ["client_id","image","cam","proj","dist","ts"])
frame_buffers = defaultdict(list)

# POST /post_image/{id} - ENHANCED VERSION
async def post_image(request):
    global image_bgr, latest_frame_data

    client_id = request.match_info["id"]
    try:
        data = await request.json()
        base64_str = data.get("image")
        if not base64_str:
            return web.Response(status=400, text="Missing image data")

        timestamp = data.get("timestamp", -1)
        camera_to_world = data.get("cameraToWorldMatrix", [])
        instrinsics = data.get("instrinsics", [])
        distortion = data.get("distortion", [])

        # Process camera matrices
        cam_mat = None
        proj_mat = None
        dist_mat = None
        camera_to_world_mat = None

        if len(camera_to_world) == 16:
            values = list(map(float, camera_to_world))
            camera_to_world_mat = np.array([values[i:i+4] for i in range(0, 16, 4)])

        if len(instrinsics) == 16:
            values = list(map(float, instrinsics))
            proj_mat = np.array([values[i:i+4] for i in range(0, 16, 4)])
            
            # Extract camera matrix from projection matrix
            cam_mat = np.array([
                [proj_mat[0][0], 0, proj_mat[0][2]],
                [0, proj_mat[1][1], proj_mat[1][2]],
                [0, 0, 1]
            ])
            
            if not board_tracker.params_initialized:
                board_tracker.assign_camera_params(proj_mat[0][0], proj_mat[1][1], proj_mat[0][2], proj_mat[1][2])

        if len(distortion) == 16:
            values = list(map(float, distortion))
            dist_mat = np.array([values[i:i+4] for i in range(0, 16, 4)])

        # Decode base64 string
        image_bytes = base64.b64decode(base64_str)
        image = Image.open(BytesIO(image_bytes))
        image = ImageEnhance.Brightness(image).enhance(0.9) # decrease brightness
        image = ImageEnhance.Contrast(image).enhance(1.5) # increase contrast
        image_rgb = np.array(image)
        image_bgr_current = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite("text.png", image_bgr_current)

        # Store in frame buffers
        frame_buffers[client_id].append(
            Frame(client_id, image_bgr_current, cam_mat, proj_mat, dist_mat, timestamp)
        )
        
        # Store for Gradio (FIXED: append the image, not overwrite)
        image_bgr.append(image_bgr_current)
        
        # Store complete frame data for annotation processing
        latest_frame_data = FrameData(
            client_id=client_id,
            image=image_bgr_current,
            camera_matrix=cam_mat,
            projection_matrix=proj_mat,
            distortion_matrix=dist_mat,
            timestamp=timestamp,
            camera_to_world=camera_to_world_mat
        )
        
        # Keep only recent frames
        if len(image_bgr) > 10:
            image_bgr.popleft()
        
        logger.debug("Received image from client %s", client_id)
        return web.Response(status=200, text="Image received successfully")

    except Exception as e:
        logger.error("Error receiving image from client %s: %s", client_id, e)
        return web.Response(status=500, text="Failed to receive image")

# GET /answer/{id}
async def get_answers_for_id(request):
    client_id = int(request.match_info["id"])
    answer = created_answers.get(client_id, {})
    return web.json_response(answer)

# GET /
async def root_redirect(request):
    raise web.HTTPFound("/client/index.html")

async def on_shutdown(app):
    logger.info("Shutting down...")

    for task in consume_tasks.values():
        task.cancel()

    for pc in pcs.values():
        for transceiver in pc.getTransceivers():
            track = transceiver.receiver.track
            if track:
                try:
                    track.stop()
                except Exception as e:
                    logger.warning(f"Failed to stop track during shutdown: {e}")
        await pc.close()

    for recorder in recorders.values():
        await recorder.stop()

    pcs.clear()
    recorders.clear()
    consume_tasks.clear()

def handle_images():
    global image_bgr, msg_queue
    while True:
        try:
            if len(image_bgr) == 0:
                time.sleep(0.1)
                continue

            frame = image_bgr[-1]
            if frame is None:
                break
            # Add any image processing logic here
            
        except Exception as e:
            logger.error(f"handle_images encountered an error: {e}", exc_info=True)
            continue

def run_server(args):
    # SSL Setup
    ssl_context = None
    if not args.no_ssl:
        try:
            ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
            ssl_context.load_cert_chain(args.pem)
        except FileNotFoundError:
            logger.warning("SSL certificate not found, running without SSL")

    app = web.Application(middlewares=[cors_middleware], client_max_size=10*1024**2)

    # Routes
    app.router.add_post("/login", login)
    app.router.add_post(r"/logout/{id:\d+}", logout)
    app.router.add_post(r"/post_offer/{id:\d+}", post_offer)
    app.router.add_post(r"/post_answer/{from_id:\d+}/{to_id:\d+}", post_answer)
    app.router.add_post(r"/post_ice/{id:\d+}", post_ice)
    app.router.add_post(r"/post_image/{id:\d+}", post_image)
    app.router.add_post(r"/consume_ices/{id:\d+}", consume_ices)
    app.router.add_get("/offers", get_offers)
    app.router.add_get("/answers", get_answers)
    app.router.add_get(r"/answer/{id:\d+}", get_answers_for_id)
    app.router.add_get("/latest-frame", get_latest_frame)
    app.router.add_post("/submit-annotation", submit_annotation)  # NEW ANNOTATION ROUTE
    app.router.add_get("/test", lambda request: web.Response(text="🟢 Server is working! Time: " + str(time.time())))  # TEST ENDPOINT
    app.router.add_get("/debug", lambda request: web.Response(text=f"Server running on {args.ip}:{args.port}\nConnected clients: {len(pcs)}\nActive frames: {len(image_bgr)}"))
    app.router.add_get("/", root_redirect)
    app.on_shutdown.append(on_shutdown)

    logger.info(f"Starting XaiR Server on {args.ip}:{args.port}")
    web.run_app(app, host=args.ip, port=args.port, access_log=None, ssl_context=ssl_context)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="XaiR Server.")
    parser.add_argument('--ip', default='127.0.0.1', help='IP address to bind to')
    parser.add_argument('--port', type=int, default=8000, help='Port to bind to')
    parser.add_argument('--no-ssl', action='store_true', help='Disable SSL')
    parser.add_argument('--pem', default='server.pem', help='Path to SSL certificate')
    parser.add_argument('-v', '--verbosity', type=int, default=2,
                        help='Set log level (0: ERROR, 1: WARNING, 2: INFO, 3: DEBUG)')
    parser.add_argument('--instruct', action='store_true', help='Enable instruction following')
    args = parser.parse_args()

    verbosity_map = {
        0: logging.ERROR,
        1: logging.WARNING,
        2: logging.INFO,
        3: logging.DEBUG,
    }
    log_level = verbosity_map.get(args.verbosity, logging.DEBUG)
    logger.setLevel(log_level)

    # Start the thread
    display_thread = Thread(target=handle_images, daemon=True)
    display_thread.start()

    run_server(args)