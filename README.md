# Aqua Survai

Laptop-side model inference server for Raspberry Pi camera monitoring.

Run only `laptop_detection_server.py` from this repository. It receives Raspberry Pi camera frames, runs the project's fixed `model1.pt`, and returns detections over the same TCP socket.

## Files

| File | Run on | Purpose |
| --- | --- | --- |
| `laptop_detection_server.py` | Laptop | The main file to run. Receives frames, runs `model1.pt`, and returns JSON detections. |
| `rpi_camera_client.py` | Backup/reference | Copy this file to a Raspberry Pi only if a Pi sender/client is needed. Do not run it as part of the laptop setup. |
| `model1.pt` | Laptop | The project's trained model. Keep it beside `laptop_detection_server.py`. |

## Network requirements

- Connect the laptop and Raspberry Pi to the same network.
- The deployed Raspberry Pi sender must connect to the laptop IP address and port below.
- The laptop listens on TCP port `5000` by default. Ensure the firewall permits it.

Find the laptop IP address with:

```bash
hostname -I
```

## Execution examples

Run the laptop server:

```bash
cd /home/couliglig1/aqua_survai
source venv1/bin/activate
python laptop_detection_server.py --port 5000 --device cpu --imgsz 416 --show
```

For faster laptop inference with lower image detail:

```bash
python laptop_detection_server.py --port 5000 --device cpu --imgsz 320 --show
```

## Laptop setup

Run these commands in the project directory:

```bash
python3 -m venv venv1
source venv1/bin/activate
pip install --upgrade pip
pip install opencv-python numpy ultralytics
```

`ultralytics` installs the model runtime and PyTorch dependencies needed to execute `model1.pt`.

Start the laptop server:

```bash
source venv1/bin/activate
python laptop_detection_server.py --port 5000 --device cpu --imgsz 416 --show
```

The model path is fixed to `model1.pt`; there is no model-selection argument.

## Raspberry Pi backup client

`rpi_camera_client.py` is retained as a backup/reference sender file. If you need to deploy or restore the Pi client, copy it to the Raspberry Pi. It is not required to run the laptop server in this repository.

The backup client requires OpenCV on the Pi:

```bash
python3 -m venv venv1
source venv1/bin/activate
pip install --upgrade pip
pip install opencv-python
```

If your Pi system Python already supplies a working `cv2`, you can use that instead of installing OpenCV in the virtual environment.

Example backup-client command, replacing the IP address with the laptop's address:

```bash
python3 rpi_camera_client.py \
  --laptop-ip 192.168.20.133 \
  --port 5000 \
  --display-window true
```

## Backup client configuration

The default Pi settings are at the top of `rpi_camera_client.py`:

```python
LAPTOP_IP = "192.168.20.133"
LAPTOP_PORT = 5000
CAMERA_WIDTH = 800
CAMERA_HEIGHT = 600
JPEG_QUALITY = 50
CONF_THRESHOLD = 0.85
```

The backup client draws the laptop's returned boxes and labels in its OpenCV preview. Press `q` or `Esc` in its preview window to stop it.

## Returned detection format

The laptop sends one JSON reply per image:

```json
{
  "ok": true,
  "frame_id": 1,
  "image": {"width": 800, "height": 600},
  "detections": [
    {
      "class": "drowning",
      "label": "drowning",
      "class_id": 0,
      "confidence": 0.91,
      "bbox": {"x1": 100, "y1": 80, "x2": 300, "y2": 400}
    }
  ]
}
```

The Pi draws these boxes and labels. `Alert: ARMED` only means the Pi's local drowning-alert timer is ready; it does not send a message, SMS, or notification.

## Troubleshooting

- `Address already in use`: another laptop server is already using port `5000`. Stop it with `Ctrl+C` before starting another.
- `Could not connect to laptop`: confirm the IP address, port, Wi-Fi connection, and firewall rule.
- No boxes: confirm the model labels and lower `CONF_THRESHOLD` on the Pi if the laptop returns detections below `0.85` confidence.
- Slow video: reduce `CAMERA_WIDTH`/`CAMERA_HEIGHT`, lower `JPEG_QUALITY`, or run the laptop server with `--imgsz 320`.
