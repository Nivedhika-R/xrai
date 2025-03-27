import ssl
import os
import time
import asyncio
import base64
from threading import Thread
from queue import Queue
from collections import deque
import logging
import argparse
import soundfile as sf
from datetime import datetime, timedelta, timezone

import cv2
from PIL import Image
from io import BytesIO
import whisper
import torch

import numpy as np

from collections import defaultdict
from aiohttp import web
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription, RTCIceCandidate
from av import AudioResampler

from constants import *
from chatgpt_helper import ChatGPTHelper

# configure logging
formatter = logging.Formatter('[%(asctime)s] [%(levelname).1s] %(message)s', datefmt='%H:%M:%S')
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(handler)

server_id = 0
next_client_id = 1
created_offers = {}
created_answers = {}
pcs = {}  # client_id: RTCPeerConnection
recorders = {}  # client_id: MediaRecorder
ices = defaultdict(list)

image_deque = deque()

chatgpt = ChatGPTHelper()
llm_reply = ""

class WhisperProcessor:
    def __init__(self, model_name="medium", record_timeout=5, phrase_timeout=3):
        self.audio_model = whisper.load_model(model_name + ".en")
        self.data_queue = Queue()
        self.transcription = [""]
        self.phrase_time = None
        self.record_timeout = record_timeout
        self.phrase_timeout = phrase_timeout

    def feed_audio(self, pcm_bytes: bytes):
        self.data_queue.put(pcm_bytes)

    def run(self):
        while True:
            now = datetime.now(timezone.utc)

            if not self.data_queue.empty():
                phrase_complete = False

                if self.phrase_time and now - self.phrase_time > timedelta(seconds=self.phrase_timeout):
                    phrase_complete = True

                self.phrase_time = now

                # Combine all audio in the queue
                audio_data = b''.join(self.data_queue.queue)
                self.data_queue.queue.clear()

                # Convert to float32 PCM [-1.0, 1.0]
                audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

                # Transcribe with Whisper
                result = self.audio_model.transcribe(audio_np, fp16=torch.cuda.is_available(), language="en")
                text = result['text'].strip()

                if phrase_complete:
                    self.transcription.append(text)
                else:
                    self.transcription[-1] = text

                if text:
                    logger.info(f"\nUser said:\n{text}\n")

            time.sleep(0.1)

class RemoteAudioToWhisper(MediaStreamTrack):
    kind = "audio"

    def __init__(self, track):
        super().__init__()
        self.track = track
        self.buffer = bytearray()

        self.whisper_processor = WhisperProcessor()
        self.whisper_thread = Thread(target=self.whisper_processor.run, daemon=True)
        self.whisper_thread.start()

        self.sample_rate = 16000 # Whisper has a sample rate of 16000
        self.resampler = AudioResampler(
            format='s16',
            layout='mono',
            rate=self.sample_rate
        )

        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.debug_file = f"debug_audio_{now}.wav"
        self.wav_writer = sf.SoundFile(
            self.debug_file, mode='w', samplerate=self.sample_rate, channels=1, subtype='PCM_16'
        )

    async def recv(self):
        frame = await self.track.recv()

        resampled_frames = self.resampler.resample(frame)
        if not isinstance(resampled_frames, list):
            resampled_frames = [resampled_frames]

        for resampled in resampled_frames:
            pcm_array = resampled.to_ndarray().astype(np.int16).flatten()
            pcm_bytes = pcm_array.tobytes()

            # Write to WAV (optional for debugging)
            # self.wav_writer.write(pcm_array)

            self.whisper_processor.feed_audio(pcm_bytes)

        return frame

    def stop(self):
        self.wav_writer.close()
        super().stop()

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

    if client_id in pcs:
        await pcs[client_id].close()
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
            audio_reader = RemoteAudioToWhisper(track)
            asyncio.ensure_future(dummy_consume(audio_reader))

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
    global image_deque

    client_id = request.match_info["id"]
    try:
        data = await request.json()
        base64_str = data.get("image")
        if not base64_str:
            return web.Response(status=400, text="Missing image data")

        timestamp = data.get("timestamp", -1)
        camera_to_world = data.get("cameraToWorldMatrix", [])
        projection_matrix = data.get("projectionMatrix", [])

        if len(camera_to_world) == 16:
            cam_mat = [camera_to_world[i:i+4] for i in range(0, 16, 4)]

        if len(projection_matrix) == 16:
            proj_mat = [projection_matrix[i:i+4] for i in range(0, 16, 4)]

        # Decode base64 string
        image_bytes = base64.b64decode(base64_str)
        image = Image.open(BytesIO(image_bytes))

        image_deque.append((client_id, image, cam_mat, proj_mat, timestamp))

        logger.info("Received image from client %s", client_id)
        return web.Response(status=200, text="Image received successfully")

    except Exception as e:
        logger.error("Error receiving image from client %s: %s", client_id, e)
        return web.Response(status=500, text="Failed to process image")

# GET /answer/{id}
async def get_answers_for_id(request):
    client_id = int(request.match_info["id"])
    answer = created_answers.get(client_id, {})
    return web.json_response(answer)

# GET /
async def root_redirect(request):
    raise web.HTTPFound("/client/index.html")

# GET /llm_response
async def get_llm_response(request):
    global llm_reply
    logger.info("LLM reply: %s", llm_reply)
    return web.Response(text=llm_reply, content_type="text/plain")

# Clean shutdown
async def on_shutdown(app):
    logger.info("Shutting down...")
    for pc in pcs.values():
        await pc.close()
    for recorder in recorders.values():
        await recorder.stop()
    pcs.clear()
    recorders.clear()

def handle_images():
    global image_deque, llm_reply

    while True:
        try:
            if len(image_deque) == 0:
                time.sleep(0.1)
                continue

            client_id, img, cam_mat, proj_mat, timestamp = image_deque[-1]
            image_deque.clear()
            if img is None:
                break

            # save image to disk
            # img_path = os.path.join("images", f"image_c{client_id}_{timestamp}.png")
            # cv2.imwrite(img_path, img)

            img = np.array(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            llm_reply = chatgpt.ask("Describe what I'm looking at.", image=img)
        except Exception as e:
            logger.error("Video processing stopped: %s", e)
            break

    cv2.destroyAllWindows()

def run_server(args):
    # SSL Setup
    ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    ssl_context.load_cert_chain(args.pem)

    os.makedirs("images", exist_ok=True)

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
    app.router.add_get("/llm_response", get_llm_response)
    app.router.add_get("/", root_redirect)
    app.on_shutdown.append(on_shutdown)

    web.run_app(app, host=args.ip, port=args.port, ssl_context=ssl_context if not args.no_ssl else None)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="XaiR Server.")
    parser.add_argument('--ip', default='127.0.0.1', help='IP address to bind to')
    parser.add_argument('--port', type=int, default=8000, help='Port to bind to')
    parser.add_argument('--no-ssl', action='store_true', help='Disable SSL')
    parser.add_argument('--pem', default='server.pem', help='Path to SSL certificate')

    args = parser.parse_args()

    # Start the thread
    display_thread = Thread(target=handle_images, daemon=True)
    display_thread.start()

    run_server(args)
