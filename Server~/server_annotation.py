# Run: python server_annotation.py --ip <ip address> --port 8000 --pem server.pem

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
image_bgr = deque()

# chatgpt = ChatGPTHelper()
board_tracker = BoardTracker()
# yolo = None

#tutorial_follower = None

# llm_reply = None

# llm_images = []

debugger = None
latest_pose_data = {}
latest_annotations = []
annotation_queue = []

@web.middleware
async def cors_middleware(request, handler):
    if request.method == "OPTIONS":
        return web.Response(status=200)
    response = await handler(request)
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "*"
    return response

async def get_latest_frame(request):
    global image_bgr, latest_pose_data
    if len(image_bgr) == 0:
        return web.json_response({"image": None, "pose_matrix": None})

    frame = image_bgr[-1]  # Get the latest frame from deque
    logger.debug("Sending latest frame to client")

    # Encode the frame properly
    _, buffer = cv2.imencode('.jpg', frame)
    frame_base64 = base64.b64encode(buffer).decode('utf-8')

    # Include pose matrix in response
    response_data = {
        "image": frame_base64,
        "pose_matrix": latest_pose_data.get("camera_to_world", None) if latest_pose_data else None
    }

    return web.json_response(response_data)

async def login(request):
    global next_client_id
    client_id = next_client_id
    next_client_id += 1
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
            # audio_reader = RemoteAudioToWhisper(track)
            # consume_task = asyncio.create_task(dummy_consume(audio_reader))
            # consume_tasks[client_id] = consume_task

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

# POST /post_image/{id}

async def post_image(request):
    global image_bgr, latest_pose_data

    client_id = request.match_info["id"]
    try:
        data = await request.json()

        logger.info("🔍 DEBUG: Incoming data keys from Unity:")
        logger.info(f"   Keys: {list(data.keys())}")

        base64_str = data.get("image")
        if not base64_str:
            return web.Response(status=400, text="Missing image data")

        timestamp = data.get("timestamp", -1)
        camera_to_world = data.get("cameraToWorldMatrix", [])
        instrinsics = data.get("instrinsics", [])
        distortion = data.get("distortion", [])

        # 🔍 DEBUG: Check pose matrix data
        logger.info("🔍 DEBUG: Pose matrix data from Unity:")
        logger.info(f"   cameraToWorldMatrix length: {len(camera_to_world)}")
        logger.info(f"   instrinsics length: {len(instrinsics)}")
        logger.info(f"   distortion length: {len(distortion)}")

        if len(camera_to_world) == 16:
            logger.info("✅ Camera-to-world matrix received (16 elements)")
            logger.info(f"   First 4 elements: {camera_to_world[:4]}")
        else:
            logger.warning("❌ Camera-to-world matrix MISSING or wrong size")

        if len(instrinsics) == 16:
            logger.info("✅ Intrinsics matrix received (16 elements)")
        else:
            logger.warning("❌ Intrinsics matrix MISSING or wrong size")

        # Store pose data globally
        latest_pose_data = {
            "camera_to_world": camera_to_world,
            "intrinsics": instrinsics,
            "distortion": distortion,
            "timestamp": timestamp
        }

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

        # Decode base64 string
        image_bytes = base64.b64decode(base64_str)
        image = Image.open(BytesIO(image_bytes))
        image = ImageEnhance.Brightness(image).enhance(0.9) # decrease brightness
        image = ImageEnhance.Contrast(image).enhance(1.5) # increase contrast
        image_rgb = np.array(image)
        image_bgr_current = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        # Save to file for debugging
        cv2.imwrite("test.png", image_bgr_current)

        # Update the deque properly - ADD to deque, don't replace it
        image_bgr.append(image_bgr_current)

        # Keep only the latest 10 frames to prevent memory issues
        if len(image_bgr) > 10:
            image_bgr.popleft()

        logger.debug("Received image from client %s", client_id)
        return web.Response(status=200, text="Image received successfully")

    except Exception as e:
        logger.error("Error receiving image from client %s: %s", client_id, e)
        return web.Response(status=500, text="Failed to receive image")

# POST /submit-annotation

async def submit_annotation(request):
    global latest_annotations, annotation_queue, latest_pose_data

    try:
        data = await request.json()
        base64_str = data.get("image")
        coordinates = data.get("coordinates", [])
        pose_matrix = data.get("pose_matrix", None)
        timestamp = data.get("timestamp", int(time.time()))

        if not base64_str:
            return web.Response(status=400, text="Missing image data")

        if not coordinates:
            logger.warning("⚠️  No annotation coordinates received")
            return web.json_response({
                "status": "warning",
                "message": "No annotations found in the image"
            })

        # Decode and save the annotated image
        image_bytes = base64.b64decode(base64_str)
        image = Image.open(BytesIO(image_bytes))
        image_rgb = np.array(image)

        # Save annotated image
        filename = f"received_annotation_{timestamp}.png"
        cv2.imwrite(filename, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))

        # Process coordinates
        if coordinates:
            x_coords = [c[0] for c in coordinates]
            y_coords = [c[1] for c in coordinates]

            min_x, max_x = min(x_coords), max(x_coords)
            min_y, max_y = min(y_coords), max(y_coords)
            center_x = (min_x + max_x) // 2
            center_y = (min_y + max_y) // 2

            # 🎯 STORE ANNOTATION DATA FOR UNITY
            annotation_data = {
                "id": timestamp,
                "coordinates": coordinates,
                "center": [center_x, center_y],
                "bounding_box": {
                    "min": [min_x, min_y],
                    "max": [max_x, max_y],
                    "width": max_x - min_x,
                    "height": max_y - min_y
                },
                "pose_matrix": pose_matrix,
                "timestamp": timestamp,
                "processed": False
            }

            # Store as latest annotation
            latest_annotations = [annotation_data]

            # Add to queue for Unity to consume
            annotation_queue.append(annotation_data)

            # Keep only last 10 annotations in queue
            if len(annotation_queue) > 10:
                annotation_queue.pop(0)

            logger.info("🎯 ANNOTATION STORED FOR UNITY")
            logger.info(f"📊 Center: ({center_x}, {center_y})")
            logger.info(f"📐 Pose matrix: {'Available' if pose_matrix else 'Not available'}")
            logger.info(f"📤 Queued for Unity: {len(annotation_queue)} annotations pending")

            # # Enhanced logging
            # logger.info("=" * 50)
            # logger.info("🎯 ANNOTATION RECEIVED")
            # logger.info("=" * 50)
            # logger.info(f"📁 Saved as: {filename}")
            # logger.info(f"📊 Total annotation points: {len(coordinates)}")
            # logger.info(f"🎯 CENTER: ({center_x}, {center_y})")
            # logger.info(f"📦 BOUNDING BOX:")
            # logger.info(f"   Top-left: ({min_x}, {min_y})")
            # logger.info(f"   Bottom-right: ({max_x}, {max_y})")
            # logger.info(f"   Size: {max_x - min_x} x {max_y - min_y} pixels")

            # if pose_matrix:
            #     logger.info(f"📐 POSE MATRIX: Included ({len(pose_matrix)} elements)")
            #     # You can process the pose matrix here for your specific needs
            # else:
            #     logger.info("📐 POSE MATRIX: Not available")

            # logger.info("=" * 50)

            # # Here you can add your specific processing logic
            # # For example, convert image coordinates to world coordinates using pose matrix

        return web.json_response({
            "status": "success",
            "total_points": len(coordinates),
            "center": [center_x, center_y] if coordinates else None,
            "pose_matrix_received": pose_matrix is not None,
            "message": f"Successfully processed annotation with {len(coordinates)} points",
            "saved_as": filename,
            "annotation_id": timestamp
        })

    except Exception as e:
        logger.error("❌ Error processing annotation: %s", e)
        import traceback
        traceback.print_exc()
        return web.Response(status=500, text=f"Failed to process annotation: {e}")

async def get_latest_annotation(request):
    """Get only the most recent annotation"""
    global latest_annotations

    try:
        if not latest_annotations:
            return web.json_response({
                "status": "no_annotations",
                "annotation": None
            })

        latest = latest_annotations[-1]

        if latest.get("processed", False):
            return web.json_response({
                "status": "already_processed",
                "annotation": None
            })

        logger.info(f"📤 Sending latest annotation (ID: {latest['id']}) to Unity")

        return web.json_response({
            "status": "success",
            "annotation": latest
        })

    except Exception as e:
        logger.error("❌ Error getting latest annotation: %s", e)
        return web.Response(status=500, text=f"Failed to get latest annotation: {e}")

async def mark_annotations_processed(request):
    """Mark annotations as processed by Unity"""
    global annotation_queue, latest_annotations

    try:
        data = await request.json()
        processed_ids = data.get("processed_ids", [])

        # Mark annotations as processed in both queues
        for annotation in annotation_queue:
            if annotation["id"] in processed_ids:
                annotation["processed"] = True

        for annotation in latest_annotations:
            if annotation["id"] in processed_ids:
                annotation["processed"] = True

        logger.info(f"✅ Marked {len(processed_ids)} annotations as processed by Unity")

        return web.json_response({
            "status": "success",
            "processed_count": len(processed_ids)
        })

    except Exception as e:
        logger.error("❌ Error marking annotations as processed: %s", e)
        return web.Response(status=500, text=f"Failed to mark annotations as processed: {e}")

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

def run_server(args):
    # SSL Setup
    ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    ssl_context.load_cert_chain(args.pem)

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
    app.router.add_post("/submit-annotation", submit_annotation)
    app.router.add_get("/get-latest-annotation", get_latest_annotation)
    app.router.add_post("/mark-annotations-processed", mark_annotations_processed)
    # app.router.add_get("/llm-response", get_llm_response)
    # app.router.add_get("/llm-images", get_llm_images)
    app.router.add_get("/", root_redirect)
    app.on_shutdown.append(on_shutdown)

    web.run_app(app, host=args.ip, port=args.port, access_log=None, ssl_context=ssl_context if not args.no_ssl else None)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="XaiR Server.")
    parser.add_argument('--ip', default='127.0.0.1', help='IP address to bind to')
    parser.add_argument('--port', type=int, default=8000, help='Port to bind to')
    parser.add_argument('--no-ssl', action='store_true', help='Disable SSL')
    parser.add_argument('--pem', default='server.pem', help='Path to SSL certificate')
    parser.add_argument('-v', '--verbosity', type=int, default=2,
                        help='Set log level (0: ERROR, 1: WARNING, 2: INFO, 3: DEBUG)')
    parser.add_argument('--instruct', action='store_true', help='Enable instruction following')

    # parser.add_argument('--yolo-model', default='runs/detect/train_latest/best.pt', help='Path to YOLO model')
    args = parser.parse_args()
    verbosity_map = {
        0: logging.ERROR,
        1: logging.WARNING,
        2: logging.INFO,
        3: logging.DEBUG,
    }
    log_level = verbosity_map.get(args.verbosity, logging.DEBUG)
    logger.setLevel(log_level)
    #yolo = YoloHelper(args.yolo_model)
    # if args.instruct:
    #     tutorial_follower = TutorialFollower(image_bgr, board_tracker=board_tracker)
    #     debugger = Debugger(tutorial_follower)
    run_server(args)
