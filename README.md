# XR-Care

XR-Care is an AR application that combines Unity's AR Foundation with WebRTC communication and a web-based annotation system. The project enables real-time AR camera feed sharing, annotation creation through a web interface, and synchronized display of annotations in AR space across multiple devices.

**For complete setup guide and troubleshooting:** https://xrcare.netlify.app

## Quick Start

### Prerequisites
- Unity 6 (Latest LTS)
- Python 3.8+
- ARCore-compatible Android device
- Git (optional)

### Server Setup

1. Navigate to server directory:
```bash
cd Server~/
```

2. Create SSL certificate:
```bash
openssl req -new -x509 -keyout server.pem -out server.pem -days 365 -nodes
```

3. Install Python dependencies:
```bash
pip install -r requirements.txt
```

4. Start the main server:
```bash
python server.py --ip <YOUR_IP_ADDRESS> --port 8000 --pem server.pem
```

5. Launch preview application:
```bash
python preview.py --ip <YOUR_IP_ADDRESS>
```

### Unity Client Setup

1. Open the repository root in Unity Editor
2. Load the `XaiRScene` scene
3. Configure the `XaiRClient` component:
   - Set `Ip Address` to your server IP
   - Set `Port` to `8000`
4. Switch build target to Android
5. Connect ARCore-compatible Android device
6. Build and Run

### Development Mode

Run in Unity Editor by pressing Play (limited AR capabilities - no spatial meshing or raycast features).

## System Requirements

**Hardware:**
- ARCore-compatible Android device
- Development PC (Windows/Mac/Linux)
- USB cable for device connection
- Stable WiFi network

**Software:**
- Unity 6 (Latest LTS)
- AR Foundation package
- ARCore XR Plugin
- Python 3.8+
- Android SDK & NDK

## Technology Stack

- **Unity 6** - Core AR development platform
- **AR Foundation** - Unity's cross-platform AR framework
- **ARCore** - Google's AR platform for Android
- **WebRTC** - Real-time peer-to-peer communication
- **Python Server** - Backend coordination and web interface

## Project Structure

```
├── Assets/                 # Unity project assets
├── Server~/               # Python server components
│   ├── server.py         # Main server application
│   ├── preview.py        # Gradio web interface
│   └── requirements.txt  # Python dependencies
├── Scenes/
│   └── XaiRScene.unity   # Main AR scene
└── README.md
```

## Features

- Real-time AR camera feed sharing
- Web-based annotation creation interface
- Synchronized AR object placement across devices
- WebRTC peer-to-peer communication
- Cross-device annotation synchronization
- Spatial mapping and tracking

## Troubleshooting

For comprehensive troubleshooting guide covering Unity setup, Python server issues, Android device problems, and network configuration, visit: https://xrcare.netlify.app

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

[Add your license information here]

## Support

For detailed setup instructions, troubleshooting, and video tutorials, visit our comprehensive guide: https://xrcare.netlify.app