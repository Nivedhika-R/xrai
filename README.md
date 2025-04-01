# XaiR Development

## Server Setup
Enter server directory:
```
cd Server~/
```

Enable ssl/https by creating a key file:
```
openssl req -new -x509 -keyout server.pem -out server.pem -days 365 -nodes
```

Install Python dependencies:
```
pip install -r requirements.txt
```

Run the server:
```
python server.py --ip <SERVER IP ADDRESS> --port 8000 --pem server.pem
```

## Client Setup
Open the *repo root* in the Unity editor. Open the `XaiRScene` scene.
In the `XaiRClient` Component, change the `Ip Address` and `Port` fields to your server IP and port.

Change build target to `Android`. Connect MagicLeap 2 and `Build and Run`.

Optionally, you can run it in the Unity editor by pressing the play button, but you will not have meshing or raycacting capabilities.
