# Aqua Survai

Raspberry Pi camera monitoring with model inference on a laptop.

The Raspberry Pi captures a frame, JPEG-encodes it, and sends it to the laptop over TCP. The laptop runs the project's fixed `model1.pt`, then returns JSON detections over the same socket. The Pi shows the returned bounding boxes, labels, confidence values, and alert state in an OpenCV window.

## Files

| File | Run on | Purpose |
| --- | --- | --- |
| `laptop_detection_server.py` | Laptop | Receives frames, runs `model1.pt`, and returns JSON detections. |
| `rpi_camera_client.py` | Raspberry Pi | Captures/sends frames, receives JSON detections, and displays annotated video. |
| `model1.pt` | Laptop | The project's trained model. Keep it beside `laptop_detection_server.py`. |

## Network requirements

- Connect the laptop and Raspberry Pi to the same network.
- Set `LAPTOP_IP` in `rpi_camera_client.py`, or pass it through `--laptop-ip`.
- The laptop listens on TCP port `5000` by default. Ensure the firewall permits it.

Find the laptop IP address with:

```bash
hostname -I
```

## Execution examples

Start this first on the laptop:

```bash
cd /home/couliglig1/aqua_survai
source venv1/bin/activate
python laptop_detection_server.py --port 5000 --device cpu --imgsz 416 --show
```

Then start this on the Raspberry Pi:

```bash
cd ~/Downloads/FOREGDE
source venv1/bin/activate
python3 rpi_camera_client.py --laptop-ip 192.168.20.133 --port 5000 --display-window true
```

For a Raspberry Pi without a monitor:

```bash
python3 rpi_camera_client.py --laptop-ip 192.168.20.133 --port 5000 --display-window false
```

For faster laptop inference with lower image detail:

```bash
python laptop_detection_server.py --port 5000 --device cpu --imgsz 320 --show
```

## Laptop setup

Run these commands in the project directory:

```bash
cd /home/couliglig1/aqua_survai
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

## Raspberry Pi setup

Copy `rpi_camera_client.py` to the Raspberry Pi. Then create a Pi virtual environment and install OpenCV:

```bash
python3 -m venv venv1
source venv1/bin/activate
pip install --upgrade pip
pip install opencv-python
```

If your Pi system Python already supplies a working `cv2`, you can use that instead of installing OpenCV in the virtual environment.

Run the client, replacing the IP address with the laptop's address:

```bash
python3 rpi_camera_client.py \
  --laptop-ip 192.168.20.133 \
  --port 5000 \
  --display-window true
```

For a Pi without a monitor or desktop session:

```bash
python3 rpi_camera_client.py \
  --laptop-ip 192.168.20.133 \
  --port 5000 \
  --display-window false
```

## Configuration

The default Pi settings are at the top of `rpi_camera_client.py`:

```python
LAPTOP_IP = "192.168.20.133"
LAPTOP_PORT = 5000
CAMERA_WIDTH = 800
CAMERA_HEIGHT = 600
JPEG_QUALITY = 50
CONF_THRESHOLD = 0.85
```

The OpenCV preview uses the same camera frame dimensions. Press `q` or `Esc` in the Pi preview window to stop it.

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
