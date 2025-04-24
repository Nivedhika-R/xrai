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
from collections import deque

import cv2
from PIL import Image
from io import BytesIO

import numpy as np

from collections import defaultdict
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCIceCandidate

from constants import *
from logger import logger
from chatgpt_helper import ChatGPTHelper
from whisper_helper import RemoteAudioToWhisper
from yolo_helper import YoloHelper
from preview import Preview
from frame import Frame
from tutorial_follower import TutorialFollower


server_id = 0
next_client_id = 1
consume_tasks = {}
created_offers = {}
created_answers = {}
pcs = {}  # client_id: RTCPeerConnection
recorders = {}  # client_id: MediaRecorder
ices = defaultdict(list)

msg_queue = Queue()
frame_deque = deque()

chatgpt = ChatGPTHelper()
yolo = YoloHelper("./best.pt")
tutorial_follower = TutorialFollower(frame_deque, yolo=yolo)
# yolo = YoloHelper("yolo11n.pt")
preview = Preview()

@web.middleware
async def cors_middleware(request, handler):
    if request.method == "OPTIONS":
        return web.Response(status=200)
    response = await handler(request)
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "*"
    return response

# POST /login
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

# async def dummy_consume(track):
#     while True:
#         await track.recv()

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
    global frame_deque

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

        if len(camera_to_world) == 16:
            values = list(map(float, camera_to_world))
            cam_mat = np.array([values[i:i+4] for i in range(0, 16, 4)])

        if len(instrinsics) == 16:
            values = list(map(float, instrinsics))
            proj_mat = np.array([values[i:i+4] for i in range(0, 16, 4)])

        if len(distortion) == 16:
            values = list(map(float, distortion))
            dist_mat = np.array([values[i:i+4] for i in range(0, 16, 4)])

        # Decode base64 string
        image_bytes = base64.b64decode(base64_str)
        image = Image.open(BytesIO(image_bytes))
        image_rgb = np.array(image)
        image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB)

        frame_deque.append(Frame(client_id, image_rgb, cam_mat, proj_mat, dist_mat, timestamp))

        # logger.info("Received image from client %s", client_id)
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
    global frame_deque, msg_queue

    while True:
        try:
            if args.instruct:
                if len(frame_deque) == 0:
                    time.sleep(0.1)
                    continue

                frame = frame_deque[-1]
                run_object_detection(frame)
                ask_tutorial(frame)
            else:
                if len(frame_deque) == 0:
                    time.sleep(0.1)
                    continue

                frame = frame_deque[-1]
                frame_deque.clear()
                if frame.img is None:
                    break
                run_object_detection(frame) # run YOLO
                run_ask_chatgpt("what color is this.", frame) # ask ChatGPT
        except Exception as e:
            logger.error("Video processing stopped: %s", e)
            break

def run_ask_chatgpt(query, frame):
    # ask ChatGPT
    llm_reply = chatgpt.ask(query, image=frame.img)
    msg = {
        "clientID": frame.client_id,
        "type": "LLMReply",
        "content": llm_reply,
        "timestamp": frame.timestamp
    }
    msg_queue.put(msg)

def ask_tutorial(frame):
    tutorial_answer = tutorial_follower.get_answer()
    if tutorial_answer is None:
        return

    print(tutorial_answer)

    llm_reply = tutorial_answer
    tutorial_follower.clear_answer()
    msg = {
        "clientID": frame.client_id,
        "type": "LLMReply",
        "content": llm_reply,
        "timestamp": frame.timestamp,
    }
    msg_queue.put(msg)

def run_object_detection(frame):
    object_labels = []
    object_centers = []
    object_confidences = []
    yolo_results = yolo.predict(frame.img)
    for result in yolo_results:
        if args.instruct and result["class_name"] not in tutorial_follower.get_current_objects():
            continue
        object_labels.append(result["class_name"])
        bbox = result["bbox"]
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        object_centers.append((center_x, frame.img.shape[0] - center_y))
        object_confidences.append(result["confidence"])

    # # save image to disk (to debug)
    # os.makedirs("images", exist_ok=True)
    # img_path = os.path.join("images", f"image_c{frame.client_id}_{frame.timestamp}.png")
    # # draw bounding boxes
    # for result in yolo_results:
    #     bbox = result["bbox"]
    #     x1, y1, x2, y2 = bbox
    #     cv2.rectangle(frame.img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    #     cv2.putText(frame.img, result["class_name"], (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # # save the image
    # logger.warning("Saving image to %s", img_path)
    # cv2.imwrite(img_path, frame.img)

    if len(object_labels) == 0:
        return

    msg = {
        "clientID": frame.client_id,
        "type": "objectDetections",
        "content": {
            "labels": object_labels,
            "centers": object_centers,
            "confidences": object_confidences,
        },
        "imageWidth": frame.img.shape[1],
        "imageHeight": frame.img.shape[0],
        "timestamp": frame.timestamp,
        "extrinsics": frame.cam_mat.flatten().tolist(),
        "instrinsics": frame.proj_mat.flatten().tolist(),
        "distortion": frame.dist_mat.flatten().tolist()
    }
    msg_queue.put(msg)
    preview.render(img, yolo_results, timestamp, llm_reply, client_id)



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
    app.router.add_get("/", root_redirect)
    app.on_shutdown.append(on_shutdown)

    web.run_app(app, host=args.ip, port=args.port, ssl_context=ssl_context if not args.no_ssl else None)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="XaiR Server.")
    parser.add_argument('--ip', default='127.0.0.1', help='IP address to bind to')
    parser.add_argument('--port', type=int, default=8000, help='Port to bind to')
    parser.add_argument('--no-ssl', action='store_true', help='Disable SSL')
    parser.add_argument('--pem', default='server.pem', help='Path to SSL certificate')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--instruct', action='store_true', help='Enable instruction following')
    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)

    # Start the thread
    display_thread = Thread(target=handle_images, daemon=True)
    display_thread.start()

    # Start the tutorial follower thread
    if args.instruct:
        tutorial_thread = Thread(target=tutorial_follower.start, daemon=True)
        tutorial_thread.start()

    run_server(args)
