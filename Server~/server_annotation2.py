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
# from chatgpt_helper import ChatGPTHelper
# from whisper_helper import RemoteAudioToWhisper
# from yolo_helper import YoloHelper
from frame import Frame
# from tutorial_follower import TutorialFollower
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
# 🔧 FIX: Keep this as a deque, don't let it get overwritten
image_frames_deque = deque(maxlen=20)  # Renamed to avoid confusion
frame_counter = 0  # Add counter for debugging

# chatgpt = ChatGPTHelper()
board_tracker = BoardTracker()

# yolo = None
#tutorial_follower = None
# llm_reply = None
# llm_images = []
debugger = None

@web.middleware
async def cors_middleware(request, handler):
    if request.method == "OPTIONS":
        response = web.Response(status=200)
    else:
        try:
            response = await handler(request)
        except Exception as e:
            logger.error(f"Handler error: {e}")
            response = web.json_response({"error": "Internal server error"}, status=500)

    # Enhanced CORS headers
    response.headers.update({
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type, Authorization, X-Requested-With",
        "Access-Control-Max-Age": "86400",  # 24 hours
        "Access-Control-Allow-Credentials": "false"
    })
    return response

async def get_latest_frame(request):
    global image_frames_deque, frame_counter

    try:
        if len(image_frames_deque) == 0:
            logger.warning("⚠️  No frames available in deque")
            return web.json_response({
                "image": "",
                "status": "no_frames",
                "message": "No frames available yet",
                "frame_count": frame_counter,
                "deque_size": 0
            })

        latest_frame = image_frames_deque[-1]

        # Add validation
        if latest_frame is None or latest_frame.size == 0:
            logger.error("❌ Invalid frame data in deque")
            return web.json_response({
                "image": "",
                "status": "invalid_frame",
                "message": "Invalid frame data",
                "frame_count": frame_counter,
                "deque_size": len(image_frames_deque)
            })

        # Encode with error handling - reduce quality for faster transmission
        success, buffer = cv2.imencode('.jpg', latest_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if not success:
            logger.error("❌ Failed to encode frame to JPEG")
            return web.json_response({
                "image": "",
                "status": "encode_error",
                "message": "Failed to encode frame",
                "frame_count": frame_counter,
                "deque_size": len(image_frames_deque)
            })

        frame_base64 = base64.b64encode(buffer).decode('utf-8')

        logger.info(f"📤 Sending frame to Gradio - Frame #{frame_counter}, Deque size: {len(image_frames_deque)}, Base64 size: {len(frame_base64)}")

        response = web.json_response({
            "image": frame_base64,
            "status": "success",
            "timestamp": time.time(),
            "frame_size": len(frame_base64),
            "frame_count": frame_counter,
            "deque_size": len(image_frames_deque)
        })

        # Enhanced headers to prevent connection issues
        response.headers.update({
            'Cache-Control': 'no-store, no-cache, must-revalidate, max-age=0',
            'Pragma': 'no-cache',
            'Expires': '0',
            'Connection': 'keep-alive',
            'Keep-Alive': 'timeout=5, max=1000',
            'X-Accel-Buffering': 'no'  # Disable nginx buffering if behind nginx
        })

        return response

    except Exception as e:
        logger.error(f"❌ Error in get_latest_frame: {e}")
        import traceback
        traceback.print_exc()
        return web.json_response({
            "image": "",
            "status": "server_error",
            "message": f"Server error: {str(e)}",
            "frame_count": frame_counter,
            "deque_size": len(image_frames_deque) if 'image_frames_deque' in globals() else 0
        }, status=500)

# POST /login
async def login(request):
    global next_client_id
    client_id = next_client_id
    next_client_id += 1
    logger.info("🔐 User %s logged in from %s", client_id, request.remote)
    return web.Response(text=str(client_id))

# POST /logout/{id}
async def logout(request):
    client_id = int(request.match_info["id"])
    logger.info("👋 User %s logged out", client_id)

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

# 🔧 FIXED: POST /post_image/{id}
async def post_image(request):
    global image_frames_deque, frame_counter

    client_id = request.match_info["id"]
    try:
        data = await request.json()
        base64_str = data.get("image")
        if not base64_str:
            logger.error(f"❌ No image data received from client {client_id}")
            return web.Response(status=400, text="Missing image data")

        timestamp = data.get("timestamp", -1)
        camera_to_world = data.get("cameraToWorldMatrix", [])
        instrinsics = data.get("instrinsics", [])
        distortion = data.get("distortion", [])

        # Process camera matrices
        if len(camera_to_world) == 16:
            values = list(map(float, camera_to_world))
            cam_mat = np.array([values[i:i+4] for i in range(0, 16, 4)])

        if len(instrinsics) == 16:
            values = list(map(float, instrinsics))
            proj_mat = np.array([values[i:i+4] for i in range(0, 16, 4)])
            if not board_tracker.params_initialized:
                board_tracker.assign_camera_params(proj_mat[0][0], proj_mat[1][1], proj_mat[0][2], proj_mat[1][2])

        if len(distortion) == 16:
            values = list(map(float, distortion))
            dist_mat = np.array([values[i:i+4] for i in range(0, 16, 4)])

        # 🔧 FIXED: Decode and process image properly
        try:
            image_bytes = base64.b64decode(base64_str)
            image = Image.open(BytesIO(image_bytes))
            image = ImageEnhance.Brightness(image).enhance(0.9)
            image = ImageEnhance.Contrast(image).enhance(1.5)
            image_rgb = np.array(image)
            frame_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

            # 🔧 CRITICAL FIX: Don't overwrite the global deque!
            image_frames_deque.append(frame_bgr)
            frame_counter += 1

            logger.info(f"✅ Frame #{frame_counter} received from client {client_id} - Deque size: {len(image_frames_deque)}")

        except Exception as decode_error:
            logger.error(f"❌ Failed to decode/process image from client {client_id}: {decode_error}")
            return web.Response(status=500, text=f"Failed to process image: {decode_error}")

        # 🔧 OPTIONAL: Save to file for debugging
        try:
            file_path = "latest_frame.jpg"
            success = cv2.imwrite(file_path, frame_bgr)
            if success:
                logger.debug(f"💾 Frame saved to {file_path}")
            else:
                logger.warning(f"⚠️  Failed to save frame to {file_path}")
        except Exception as save_error:
            logger.warning(f"⚠️  Exception saving file: {save_error}")

        return web.Response(status=200, text="Image received successfully")

    except Exception as e:
        logger.error(f"❌ Error receiving image from client {client_id}: {e}")
        import traceback
        traceback.print_exc()
        return web.Response(status=500, text=f"Failed to receive image: {e}")

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

# POST /submit-annotation
async def submit_annotation(request):
    try:
        data = await request.json()
        base64_str = data.get("image")
        coordinates = data.get("coordinates", [])

        if not base64_str:
            return web.Response(status=400, text="Missing image data")

        # Decode image for processing (but don't save)
        image_bytes = base64.b64decode(base64_str)
        image = Image.open(BytesIO(image_bytes))
        image_rgb = np.array(image)

        # Process coordinates (keep this part)
        if coordinates:
            x_coords = [c[0] for c in coordinates]
            y_coords = [c[1] for c in coordinates]

            min_x, max_x = min(x_coords), max(x_coords)
            min_y, max_y = min(y_coords), max(y_coords)
            center_x = (min_x + max_x) // 2
            center_y = (min_y + max_y) // 2

            key_points = {
                "center": [center_x, center_y],
                "top_left": [min_x, min_y],
                "top_right": [max_x, min_y],
                "bottom_left": [min_x, max_y],
                "bottom_right": [max_x, max_y],
                "bounding_box": {
                    "width": max_x - min_x,
                    "height": max_y - min_y,
                    "area": (max_x - min_x) * (max_y - min_y)
                }
            }

            logger.info("=" * 40)
            logger.info("📍 ANNOTATION SUMMARY")
            logger.info("=" * 40)
            logger.info(f"📊 Total points detected: {len(coordinates)}")
            logger.info(f"🎯 CENTER: ({center_x}, {center_y})")
            logger.info(f"📦 BOUNDING BOX:")
            logger.info(f"   Top-left: ({min_x}, {min_y})")
            logger.info(f"   Bottom-right: ({max_x}, {max_y})")
            logger.info(f"   Size: {max_x - min_x} x {max_y - min_y} pixels")
            logger.info("=" * 40)

        else:
            key_points = {}
            logger.info("⚠️  No annotation coordinates detected")

        return web.json_response({
            "status": "success",
            "total_points": len(coordinates),
            "key_points": key_points,
            "message": f"Processed annotation with {len(coordinates)} points"
        })

    except Exception as e:
        logger.error("❌ Error processing annotation: %s", e)
        return web.Response(status=500, text=f"Failed to process annotation: {e}")

async def health_check(request):
    """Enhanced health check endpoint with frame info"""
    global image_frames_deque, frame_counter
    return web.json_response({
        "status": "healthy",
        "timestamp": time.time(),
        "frames_available": len(image_frames_deque),
        "total_frames_received": frame_counter,
        "server_running": True,
        "active_connections": len(pcs),
        "deque_maxlen": image_frames_deque.maxlen
    })

# 🔧 NEW: Debug endpoint to check frame flow
async def debug_frames(request):
    """Debug endpoint to check frame status"""
    global image_frames_deque, frame_counter

    frame_info = []
    for i, frame in enumerate(image_frames_deque):
        if frame is not None:
            frame_info.append({
                "index": i,
                "shape": frame.shape if hasattr(frame, 'shape') else "unknown",
                "dtype": str(frame.dtype) if hasattr(frame, 'dtype') else "unknown"
            })

    return web.json_response({
        "total_frames_received": frame_counter,
        "current_deque_size": len(image_frames_deque),
        "deque_max_size": image_frames_deque.maxlen,
        "frame_details": frame_info,
        "latest_frame_available": len(image_frames_deque) > 0
    })

def run_server(args):
    ssl_context = None
    if not args.no_ssl:
        ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        ssl_context.load_cert_chain(args.pem)

    # Basic app configuration (compatible with older aiohttp)
    app = web.Application(
        middlewares=[cors_middleware],
        client_max_size=10*1024**2
    )

    # Routes
    app.router.add_post("/login", login)
    app.router.add_post(r"/logout/{id:\d+}", logout)
    app.router.add_post(r"/post_offer/{id:\d+}", post_offer)
    app.router.add_post(r"/post_answer/{from_id:\d+}/{to_id:\d+}", post_answer)
    app.router.add_post(r"/post_ice/{id:\d+}", post_ice)
    app.router.add_post(r"/post_image/{id:\d+}", post_image)
    app.router.add_post(r"/consume_ices/{id:\d+}", consume_ices)
    app.router.add_post("/submit-annotation", submit_annotation)
    app.router.add_get("/offers", get_offers)
    app.router.add_get("/answers", get_answers)
    app.router.add_get(r"/answer/{id:\d+}", get_answers_for_id)
    app.router.add_get("/latest-frame", get_latest_frame)
    app.router.add_get("/health", health_check)
    app.router.add_get("/debug-frames", debug_frames)  # 🔧 NEW DEBUG ENDPOINT
    app.router.add_get("/", root_redirect)
    app.on_shutdown.append(on_shutdown)

    logger.info(f"🚀 Starting server on {args.ip}:{args.port} (SSL: {not args.no_ssl})")

    web.run_app(
        app,
        host=args.ip,
        port=args.port,
        access_log=None,
        ssl_context=ssl_context
    )

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

    run_server(args)
