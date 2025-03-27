from aiohttp import web
import ssl

# openssl req -new -x509 -keyout server.pem -out server.pem -days 365 -nodes

async def login_post(request):
    return web.Response(text="", status=200)

app = web.Application()
app.router.add_post('/login', login_post)

ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
ssl_context.load_cert_chain('server.pem')

web.run_app(app, port=8000, ssl_context=ssl_context)

