#!/usr/bin/env python3
"""Send Raspberry Pi camera frames and receive laptop detection JSON."""

import argparse
import json
import socket
import struct
import time
import traceback
from collections import deque

import cv2
import serial


# Laptop network configuration
LAPTOP_IP = "192.168.20.133"
LAPTOP_PORT = 5000
SOCKET_TIMEOUT = 10.0
RECONNECT_DELAY = 2.0
MAX_RESPONSE_BYTES = 1 * 1024 * 1024

# Camera configuration
CAMERA_ID = 0
CAMERA_WIDTH = 800
CAMERA_HEIGHT = 600
JPEG_QUALITY = 50

# Detection configuration
CONF_THRESHOLD = 0.0
FRAME_SKIP = 1
PRINT_EVERY_SEC = 1.0
SMOOTH_WINDOW = 10

# Alert configuration
ALERT_HOLD_SECONDS = 6.0
COOLDOWN_SECONDS = 60.0

# ESP serial configuration
ESP_SERIAL_PORT = "/dev/ttyACM0"
ESP_BAUD_RATE = 115200
SERIAL_RECONNECT_DELAY = 2.0

DEBUG = True
DEBUG_DETECTIONS = True

# Set False when the Raspberry Pi has no attached desktop/display.
DISPLAY_WINDOW = True
WINDOW_NAME = "Raspberry Pi - Laptop Detections"


def log(message):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def debug(message):
    if DEBUG:
        log(f"[DEBUG] {message}")


def is_drowning_label(name):
    name = name.lower()
    return "drown" in name


def is_swimming_label(name):
    name = name.lower()
    return "swim" in name


def close_socket(sock):
    if sock is not None:
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            sock.close()
        except OSError:
            pass


class ESPSerialSender:
    """Keep the ESP serial connection alive without stopping model processing."""

    def __init__(self, device, baud_rate):
        self.device = device
        self.baud_rate = baud_rate
        self.connection = None
        self.next_connect_attempt = 0.0

    def close(self):
        if self.connection is not None:
            try:
                self.connection.close()
            except serial.SerialException:
                pass
            self.connection = None

    def send_drowning_count(self, count):
        if self.connection is None:
            now = time.monotonic()
            if now < self.next_connect_attempt:
                return
            try:
                self.connection = serial.Serial(
                    self.device,
                    self.baud_rate,
                    timeout=1,
                    write_timeout=1,
                )
                log(f"Connected to ESP on {self.device} at {self.baud_rate} baud")
            except (OSError, serial.SerialException) as exc:
                log(f"Could not connect to ESP serial port: {exc}")
                self.next_connect_attempt = now + SERIAL_RECONNECT_DELAY
                return

        command = f"drowning_{count}\n"
        try:
            self.connection.write(command.encode("ascii"))
            self.connection.flush()
            debug(f"Sent ESP command: {command}")
        except (OSError, serial.SerialException) as exc:
            log(f"ESP serial write failed: {exc}")
            self.close()
            self.next_connect_attempt = time.monotonic() + SERIAL_RECONNECT_DELAY


def model_result_callback(result, esp_sender):
    """Process one model result and notify the ESP of its drowning count."""
    detections = result.get("detections", [])
    if not isinstance(detections, list):
        debug("Invalid detections value received; treating it as empty")
        detections = []

    drowning_count = 0
    swimming_count = 0
    detected_labels = []

    for detection in detections:
        if not isinstance(detection, dict):
            continue
        label = str(
            detection.get("class", detection.get("label", "unknown"))
        ).lower()
        try:
            confidence = float(detection.get("confidence", 0.0))
        except (TypeError, ValueError):
            continue
        if confidence < CONF_THRESHOLD:
            continue

        detected_labels.append(f"{label}:{confidence:.2f}")
        if is_drowning_label(label):
            drowning_count += 1
        elif is_swimming_label(label):
            swimming_count += 1

    esp_sender.send_drowning_count(drowning_count)
    return detections, drowning_count, swimming_count, detected_labels


def connect_to_laptop():
    log(f"Connecting to laptop {LAPTOP_IP}:{LAPTOP_PORT}...")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    sock.settimeout(SOCKET_TIMEOUT)
    try:
        sock.connect((LAPTOP_IP, LAPTOP_PORT))
    except OSError:
        close_socket(sock)
        raise
    log("Connected to laptop successfully")
    return sock


def recv_exact(sock, size):
    """Receive exactly size bytes or fail if the laptop disconnects."""
    data = bytearray()
    while len(data) < size:
        packet = sock.recv(size - len(data))
        if not packet:
            raise ConnectionError("Laptop disconnected")
        data.extend(packet)
    return bytes(data)


def send_frame_and_get_result(sock, frame):
    """Send one JPEG, then receive one length-prefixed JSON response."""
    encode_started = time.perf_counter()
    success, encoded_frame = cv2.imencode(
        ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
    )
    if not success:
        raise RuntimeError("Failed to encode camera frame")

    frame_bytes = encoded_frame.tobytes()
    sock.sendall(struct.pack("!I", len(frame_bytes)) + frame_bytes)

    response_size = struct.unpack("!I", recv_exact(sock, 4))[0]
    if response_size == 0 or response_size > MAX_RESPONSE_BYTES:
        raise ValueError(f"Invalid laptop response size: {response_size}")

    response_data = recv_exact(sock, response_size)
    result = json.loads(response_data.decode("utf-8"))
    if not isinstance(result, dict):
        raise ValueError("Laptop response must be a JSON object")

    round_trip_ms = (time.perf_counter() - encode_started) * 1000
    # This field is local to the Pi. It is not sent back to the laptop.
    result["_round_trip_ms"] = round_trip_ms
    # debug(
    #     f"Laptop reply frame_id={result.get('frame_id', '?')} "
    #     f"bytes_sent={len(frame_bytes)} round_trip_ms={round_trip_ms:.1f}"
    # )

    if result.get("ok") is False:
        log(f"Laptop detection error: {result.get('error', 'unknown error')}")
    return result


def open_camera():
    camera = cv2.VideoCapture(CAMERA_ID)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError(f"Could not open camera {CAMERA_ID}")
    return camera


def parse_args():
    def parse_bool(value):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y", "on"}:
            return True
        if normalized in {"false", "0", "no", "n", "off"}:
            return False
        raise argparse.ArgumentTypeError("use true or false")

    parser = argparse.ArgumentParser(
        description="Send Pi camera images to the laptop and show returned detections."
    )
    parser.add_argument(
        "--laptop-ip",
        default=LAPTOP_IP,
        help=f"laptop IP address (default: {LAPTOP_IP})",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=LAPTOP_PORT,
        help=f"laptop TCP port (default: {LAPTOP_PORT})",
    )
    parser.add_argument(
        "--display-window",
        type=parse_bool,
        default=DISPLAY_WINDOW,
        metavar="{true,false}",
        help=(
            "show the annotated OpenCV camera window "
            f"(default: {str(DISPLAY_WINDOW).lower()})"
        ),
    )
    parser.add_argument(
        "--serial-port",
        default=ESP_SERIAL_PORT,
        help=f"ESP serial device (default: {ESP_SERIAL_PORT})",
    )
    parser.add_argument(
        "--baud-rate",
        type=int,
        default=ESP_BAUD_RATE,
        help=f"ESP serial baud rate (default: {ESP_BAUD_RATE})",
    )
    return parser.parse_args()


def detection_color(label):
    """Return OpenCV BGR color based on the model's returned class name."""
    if is_drowning_label(label):
        return (0, 0, 255)  # red
    if is_swimming_label(label):
        return (0, 200, 0)  # green
    return (0, 180, 255)  # orange


def draw_detections(frame, detections, result, alert_armed):
    """Draw laptop-returned bounding boxes and status information on a frame."""
    view = frame.copy()
    height, width = view.shape[:2]
    visible_detections = 0

    for detection in detections:
        if not isinstance(detection, dict):
            continue
        label = str(detection.get("class", detection.get("label", "unknown")))
        try:
            confidence = float(detection.get("confidence", 0.0))
            bbox = detection["bbox"]
            x1 = max(0, min(width - 1, round(float(bbox["x1"]))))
            y1 = max(0, min(height - 1, round(float(bbox["y1"]))))
            x2 = max(0, min(width - 1, round(float(bbox["x2"]))))
            y2 = max(0, min(height - 1, round(float(bbox["y2"]))))
        except (KeyError, TypeError, ValueError):
            continue
        if x2 <= x1 or y2 <= y1:
            continue

        color = detection_color(label)
        cv2.rectangle(view, (x1, y1), (x2, y2), color, 2)
        text = f"{label} {confidence:.0%}"
        (text_width, text_height), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
        )
        text_y = max(text_height + 6, y1)
        cv2.rectangle(
            view,
            (x1, text_y - text_height - 6),
            (x1 + text_width + 6, text_y + baseline),
            color,
            -1,
        )
        cv2.putText(
            view,
            text,
            (x1 + 3, text_y - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        visible_detections += 1

    round_trip_ms = float(result.get("_round_trip_ms", 0.0))
    fps = 1000.0 / round_trip_ms if round_trip_ms > 0 else 0.0
    status = (
        f"Laptop: {'OK' if result.get('ok', True) else 'ERROR'} | "
        f"Detections: {visible_detections} | FPS: {fps:.1f} | "
        f"Round trip: {round_trip_ms:.0f} ms"
    )
    cv2.rectangle(view, (0, 0), (width, 30), (25, 25, 25), -1)
    cv2.putText(
        view,
        status,
        (8, 21),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    alert_text = f"Alert: {'ARMED' if alert_armed else 'DISARMED'}"
    cv2.putText(
        view,
        alert_text,
        (8, height - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0) if alert_armed else (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    return view


def main():
    global LAPTOP_IP, LAPTOP_PORT, DISPLAY_WINDOW
    args = parse_args()
    LAPTOP_IP = args.laptop_ip
    LAPTOP_PORT = args.port
    DISPLAY_WINDOW = args.display_window

    log("Starting Raspberry Pi camera client")
    log(f"Laptop={LAPTOP_IP}:{LAPTOP_PORT}, camera={CAMERA_WIDTH}x{CAMERA_HEIGHT}")

    try:
        camera = open_camera()
    except Exception as exc:
        log(f"ERROR: {exc}")
        return

    if DISPLAY_WINDOW:
        # AUTOSIZE displays the received 640x480 frame as-is and does not
        # provide the resizable/zoomable window behaviour of WINDOW_NORMAL.
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)

    sock = None
    frame_idx = 0
    last_print = time.time()
    drowning_since = None
    last_alert_time = 0.0
    alert_armed = True
    esp_sender = ESPSerialSender(args.serial_port, args.baud_rate)

    total_buf = deque(maxlen=SMOOTH_WINDOW)
    drown_buf = deque(maxlen=SMOOTH_WINDOW)
    swim_buf = deque(maxlen=SMOOTH_WINDOW)

    try:
        while True:
            ret, frame = camera.read()
            if not ret:
                log("ERROR: Failed to read camera frame")
                time.sleep(0.1)
                continue

            frame_idx += 1
            if FRAME_SKIP > 1 and frame_idx % FRAME_SKIP != 0:
                continue

            if sock is None:
                try:
                    sock = connect_to_laptop()
                except OSError as exc:
                    log(f"Could not connect to laptop: {exc}")
                    time.sleep(RECONNECT_DELAY)
                    continue

            try:
                result = send_frame_and_get_result(sock, frame)
            except (OSError, ConnectionError, ValueError, json.JSONDecodeError) as exc:
                log(f"Laptop communication failed: {exc}")
                close_socket(sock)
                sock = None
                time.sleep(RECONNECT_DELAY)
                continue

            (
                detections,
                drowning_count,
                swimming_count,
                detected_labels,
            ) = model_result_callback(result, esp_sender)

            if DEBUG_DETECTIONS and detected_labels:
                debug(f"Frame {frame_idx}: detections={detected_labels}")

            total_count = drowning_count + swimming_count
            drown_buf.append(drowning_count)
            swim_buf.append(swimming_count)
            total_buf.append(total_count)
            now = time.time()

            if drowning_count > 0:
                if drowning_since is None:
                    drowning_since = now
                    log(
                        "DROWNING DETECTED - starting "
                        f"{ALERT_HOLD_SECONDS:.1f}s timer"
                    )
                elif DEBUG:
                    debug(
                        "Drowning continuing: "
                        f"{now - drowning_since:.1f}/{ALERT_HOLD_SECONDS:.1f}s"
                    )

                if (
                    alert_armed
                    and drowning_since is not None
                    and now - drowning_since >= ALERT_HOLD_SECONDS
                ):
                    log("!!! DROWNING ALERT !!!")
                    log(
                        f"{total_count} Person(s) Detected. "
                        f"{drowning_count} Detected Drowning"
                    )
                    last_alert_time = now
                    alert_armed = False
            else:
                if drowning_since is not None:
                    debug(f"Drowning cleared after {now - drowning_since:.1f}s")
                    drowning_since = None
                if not alert_armed and now - last_alert_time >= COOLDOWN_SECONDS:
                    alert_armed = True
                    log("Alert re-armed after cooldown")

            if now - last_print >= PRINT_EVERY_SEC:
                avg_total = round(sum(total_buf) / max(1, len(total_buf)))
                avg_drown = round(sum(drown_buf) / max(1, len(drown_buf)))
                avg_swim = round(sum(swim_buf) / max(1, len(swim_buf)))
                print(
                    f"[{time.strftime('%H:%M:%S')}] "
                    f"total:{avg_total} drowning:{avg_drown} "
                    f"swimming:{avg_swim} "
                    f"alert:{'ARMED' if alert_armed else 'DISARMED'} "
                    "server:CONNECTED",
                    flush=True,
                )
                last_print = now

            if DISPLAY_WINDOW:
                display_frame = draw_detections(
                    frame, detections, result, alert_armed
                )
                cv2.imshow(WINDOW_NAME, display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    log("Preview closed by user")
                    return

    except KeyboardInterrupt:
        log("Keyboard interrupt received")
    except Exception as exc:
        log(f"Streaming error: {exc}")
        if DEBUG:
            traceback.print_exc()
    finally:
        log("Closing camera and socket")
        camera.release()
        close_socket(sock)
        esp_sender.close()
        if DISPLAY_WINDOW:
            cv2.destroyAllWindows()
        log("Raspberry Pi camera client stopped")


if __name__ == "__main__":
    main()
