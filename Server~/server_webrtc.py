import ssl
import json
import asyncio
import logging
from collections import defaultdict
from aiohttp import web
from aiortc import (
    RTCPeerConnection,
    RTCSessionDescription,
    RTCIceCandidate,
)
from av import VideoFrame
from aiortc.contrib.media import MediaRecorder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("server")

server_id = 0
next_client_id = 1
created_offers = {}
created_answers = {}
pcs = {}  # client_id: RTCPeerConnection
recorders = {}  # client_id: MediaRecorder
ices = defaultdict(list)

@web.middleware
async def cors_middleware(request, handler):
    if request.method == "OPTIONS":
        return web.Response(status=200)
    response = await handler(request)
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "*"
    return response

app = web.Application(middlewares=[cors_middleware])

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
            logger.info("Receiving video from client")

            async def read_frames():
                while True:
                    try:
                        frame: VideoFrame = await track.recv()
                        img: Image.Image = frame.to_image()

                        # Save frame to disk (you can change this to your use-case)
                        img_path = os.path.join("./images", f"frame_{frame.pts}.png")
                        img.save(img_path)
                        logger.info("Saved frame to %s", img_path)

                    except Exception as e:
                        logger.warning("Error reading video frame: %s", e)
                        break

            # Run frame processing in background
            asyncio.create_task(read_frames())

        else:
            logger.info("Received non-video track: %s", track.kind)

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

        # Store answer in the exact format Unity expects
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

# GET /answer/{id}
async def get_answers_for_id(request):
    client_id = int(request.match_info["id"])
    answer = created_answers.get(client_id, {})
    print(created_answers)
    return web.json_response(answer)

# GET /
async def root_redirect(request):
    raise web.HTTPFound("/client/index.html")

# GET /client/{filename}
async def client_file(request):
    filename = request.match_info["filename"]
    return web.FileResponse(path=f"../Browser/{filename}")

# Clean shutdown
async def on_shutdown(app):
    logger.info("Shutting down...")
    for pc in pcs.values():
        await pc.close()
    for recorder in recorders.values():
        await recorder.stop()
    pcs.clear()
    recorders.clear()

# Routes
app.router.add_post("/login", login)
app.router.add_post(r"/logout/{id:\d+}", logout)
app.router.add_post(r"/post_offer/{id:\d+}", post_offer)
app.router.add_post(r"/post_answer/{from_id:\d+}/{to_id:\d+}", post_answer)
app.router.add_post(r"/post_ice/{id:\d+}", post_ice)
app.router.add_post(r"/consume_ices/{id:\d+}", consume_ices)
app.router.add_get("/offers", get_offers)
app.router.add_get("/answers", get_answers)
app.router.add_get(r"/answer/{id:\d+}", get_answers_for_id)
app.router.add_get("/client/{filename}", client_file)
app.router.add_get("/", root_redirect)

app.on_shutdown.append(on_shutdown)

# SSL Setup
ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
ssl_context.load_cert_chain("server.pem")

# Run app
if __name__ == "__main__":
    import os
    import sys

    os.makedirs("recordings", exist_ok=True)
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    web.run_app(app, host="0.0.0.0", port=port, ssl_context=ssl_context)
