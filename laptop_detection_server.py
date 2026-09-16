#!/usr/bin/env python3
"""Receive camera frames, detect objects, and reply with JSON on one TCP socket.

Wire protocol, repeated for every frame:

    Raspberry Pi -> laptop: 4-byte big-endian JPEG length, then JPEG bytes
    laptop -> Raspberry Pi:  4-byte big-endian JSON length, then UTF-8 JSON bytes

The Raspberry Pi must wait for the JSON reply before sending the next frame.
"""

from __future__ import annotations

import argparse
import json
import logging
import socket
import struct
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np


HEADER = struct.Struct("!I")
MAX_FRAME_BYTES = 20 * 1024 * 1024
MODEL_PATH = Path(__file__).with_name("model1.pt")
LOGGER = logging.getLogger("image_receiver")


def receive_exact(sock: socket.socket, size: int) -> bytes:
    """Receive exactly *size* bytes or raise ConnectionError."""
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError("peer closed the connection")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def receive_frame(sock: socket.socket) -> tuple[np.ndarray, bytes]:
    (frame_size,) = HEADER.unpack(receive_exact(sock, HEADER.size))
    if frame_size == 0 or frame_size > MAX_FRAME_BYTES:
        raise ValueError(f"invalid frame size: {frame_size} bytes")

    jpeg_bytes = receive_exact(sock, frame_size)
    encoded = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("received data is not a valid JPEG image")
    return frame, jpeg_bytes


def send_json(sock: socket.socket, message: dict[str, Any]) -> None:
    payload = json.dumps(message, separators=(",", ":")).encode("utf-8")
    sock.sendall(HEADER.pack(len(payload)) + payload)


class CustomProjectModel:
    """Inference wrapper for this project's fixed model1.pt model."""

    def __init__(
        self,
        confidence: float,
        image_size: int,
        device: str | None,
        half: bool,
        max_detections: int,
    ) -> None:
        self.confidence = confidence
        self.image_size = image_size
        self.device = device
        self.half = half
        self.max_detections = max_detections
        if not MODEL_PATH.is_file():
            raise SystemExit(f"Required project model is missing: {MODEL_PATH}")
        try:
            # model1.pt is saved in this runtime's .pt format. This is only the
            # loader needed to execute our fixed, custom-trained project model.
            from ultralytics import YOLO as ProjectModelRuntime
        except ImportError as exc:
            raise SystemExit(
                "The model1.pt runtime is missing. Install it with: "
                "pip install ultralytics"
            ) from exc
        self.model = ProjectModelRuntime(str(MODEL_PATH))
        LOGGER.info(
            "Loaded project model=%s imgsz=%d device=%s half=%s max_det=%d",
            MODEL_PATH.name,
            image_size,
            device or "auto",
            half,
            max_detections,
        )

    def detect(self, frame: np.ndarray) -> list[dict[str, Any]]:
        predict_options: dict[str, Any] = {
            "conf": self.confidence,
            "imgsz": self.image_size,
            "device": self.device,
            "max_det": self.max_detections,
            "verbose": False,
        }
        if self.half:
            predict_options["half"] = True
        result = self.model.predict(frame, **predict_options)[0]
        detections: list[dict[str, Any]] = []
        # Make one device-to-CPU transfer instead of synchronizing once per box.
        boxes_xyxy = result.boxes.xyxy.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        confidences = result.boxes.conf.cpu().numpy()
        for coordinates, class_id, confidence in zip(
            boxes_xyxy, class_ids, confidences
        ):
            x1, y1, x2, y2 = (round(float(value), 2) for value in coordinates)
            label = str(result.names[class_id])
            detections.append(
                {
                    "class_id": int(class_id),
                    # The Raspberry Pi sender consumes "class". Keep "label"
                    # too for compatibility with other clients.
                    "class": label,
                    "label": label,
                    "confidence": round(float(confidence), 4),
                    "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                }
            )
        return detections


def detection_color(label: str) -> tuple[int, int, int]:
    """Return an OpenCV BGR color for the model's class name."""
    normalized = label.lower()
    if "drown" in normalized:
        return (0, 0, 255)  # red
    if "swim" in normalized:
        return (0, 200, 0)  # green
    return (0, 180, 255)  # orange


def draw_preview(
    frame: np.ndarray, response: dict[str, Any], fps: float
) -> np.ndarray:
    """Draw local preview overlays without changing the network response."""
    view = frame.copy()
    height, width = view.shape[:2]
    visible_detections = 0

    for detection in response.get("detections", []):
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

    status = (
        f"Model: {'OK' if response.get('ok') else 'ERROR'} | "
        f"Detections: {visible_detections} | "
        f"FPS: {fps:.1f} | "
        f"Processing: {response.get('processing_ms', 0):.0f} ms"
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
    return view


def serve_client(
    client: socket.socket,
    detector: CustomProjectModel,
    show: bool,
) -> None:
    frame_id = 0
    previous_completed_at: float | None = None
    while True:
        cycle_started = time.perf_counter()
        receive_started = time.perf_counter()
        frame, jpeg_bytes = receive_frame(client)
        frame_id += 1
        receive_ms = (time.perf_counter() - receive_started) * 1000
        # LOGGER.info(
        #     "Received frame=%d bytes=%d resolution=%dx%d receive_ms=%.2f",
        #     frame_id,
        #     len(jpeg_bytes),
        #     frame.shape[1],
        #     frame.shape[0],
        #     receive_ms,
        # )

        started = time.perf_counter()

        try:
            detections = detector.detect(frame)
            response: dict[str, Any] = {
                "ok": True,
                "frame_id": frame_id,
                "image": {"width": frame.shape[1], "height": frame.shape[0]},
                "detections": detections,
                "processing_ms": round((time.perf_counter() - started) * 1000, 2),
            }
        except Exception as exc:
            LOGGER.exception("Detection failed for frame=%d", frame_id)
            response = {
                "ok": False,
                "frame_id": frame_id,
                "error": str(exc),
                "detections": [],
            }

        send_json(client, response)
        completed_at = time.perf_counter()
        cycle_ms = (completed_at - cycle_started) * 1000
        fps = (
            1.0 / (completed_at - previous_completed_at)
            if previous_completed_at is not None
            else 0.0
        )
        previous_completed_at = completed_at
        LOGGER.info(
            "Sent response frame=%d detections=%d ok=%s cycle_ms=%.2f fps=%.2f",
            frame_id,
            len(response["detections"]),
            response["ok"],
            cycle_ms,
            fps,
        )
        if show:
            cv2.imshow("Raspberry Pi Camera", draw_preview(frame, response, fps))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                LOGGER.info("Preview closed by user")
                raise KeyboardInterrupt


def run_server(
    host: str,
    port: int,
    detector: CustomProjectModel,
    show: bool,
) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((host, port))
        server.listen(1)
        LOGGER.info("Listening on %s:%d", host, port)

        while True:
            client, address = server.accept()
            LOGGER.info("Raspberry Pi connected from %s:%d", address[0], address[1])
            with client:
                client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                client.settimeout(30.0)
                try:
                    serve_client(client, detector, show)
                except (ConnectionError, ConnectionResetError, BrokenPipeError) as exc:
                    LOGGER.warning("Client disconnected: %s", exc)
                except (OSError, ValueError) as exc:
                    LOGGER.error("Connection error: %s", exc)


def configure_logging() -> None:
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    LOGGER.addHandler(console_handler)
    LOGGER.propagate = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0", help="address to listen on")
    parser.add_argument("--port", type=int, default=5000, help="TCP port")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument(
        "--imgsz",
        type=int,
        default=416,
        help="model inference size; 416 is faster, 640 is more accurate",
    )
    parser.add_argument(
        "--device",
        help="inference device, for example 0 for the first GPU or cpu",
    )
    parser.add_argument(
        "--half",
        action="store_true",
        help="use FP16 inference (GPU only)",
    )
    parser.add_argument(
        "--max-det",
        type=int,
        default=100,
        help="maximum detections returned per image",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="show received images in a live OpenCV window; press q or Esc to quit",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging()
    detector = CustomProjectModel(
        args.confidence,
        args.imgsz,
        args.device,
        args.half,
        args.max_det,
    )
    try:
        run_server(args.host, args.port, detector, args.show)
    except KeyboardInterrupt:
        LOGGER.info("Receiver stopped")
    finally:
        if args.show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
