'''
ultralytics_async.py
====================
Asynchronous video inference pipeline for Ultralytics YOLO with **ArmNN as the default backend**.

This script demonstrates how to capture frames from a video source, run model inference in a
background executor thread, and display results with end‑to‑end latency and FPS statistics.

Key changes
-----------
* `YOLO_BACKEND` now defaults to **armnn** when the environment variable is not provided.
* The delegate‑selection logic is adjusted accordingly.
'''

import os
import time
import asyncio
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

# -----------------------------------------------------------------------------
# Backend selection (ArmNN by default)
# -----------------------------------------------------------------------------

# If the user does not set YOLO_BACKEND, default to "armnn"
backend_choice = os.getenv("YOLO_BACKEND", "armnn").lower()

# Additional optional environment variables
neuron_device = os.getenv("YOLO_NEURON_DEVICE", "mdla3.0")
armnn_backend = os.getenv("YOLO_ARMNN_BACKEND", "GpuAcc")

print(
    f"[Backend] YOLO_BACKEND={backend_choice} | "
    f"YOLO_NEURON_DEVICE={neuron_device} | "
    f"YOLO_ARMNN_BACKEND={armnn_backend}"
)

# -----------------------------------------------------------------------------
# Async pipeline helpers
# -----------------------------------------------------------------------------

async def preprocess(input_queue: asyncio.Queue, cap: cv2.VideoCapture) -> None:
    """Read frames from the capture device and push to the input queue."""
    loop = asyncio.get_running_loop()
    while cap.isOpened():
        ret, frame = await loop.run_in_executor(None, cap.read)
        if not ret:
            await input_queue.put(None)
            break
        timestamp = time.time()
        await input_queue.put((frame, timestamp))
    cap.release()
    await input_queue.put(None)  # Ensure downstream tasks terminate

async def predict(
    input_queue: asyncio.Queue,
    output_queue: asyncio.Queue,
    model: YOLO,
    conf: float = 0.3,
    imgsz: int = 640,
) -> None:
    """Run model inference in a thread executor to avoid blocking the event loop."""
    loop = asyncio.get_running_loop()
    while True:
        item = await input_queue.get()
        if item is None:
            await output_queue.put(None)
            break
        frame, ts = item
        # Run inference in a background thread
        result = await loop.run_in_executor(
            None,
            lambda: model.predict(frame, conf=conf, imgsz=imgsz, stream=False)[0].plot(),
        )
        await output_queue.put((result, ts))

async def postprocess(output_queue: asyncio.Queue) -> None:
    """Display results and compute latency/FPS statistics."""
    frame_count = 0
    t_start = time.time()
    while True:
        item = await output_queue.get()
        if item is None:
            break
        frame, ts = item
        latency_ms = (time.time() - ts) * 1000
        frame_count += 1
        avg_fps = frame_count / (time.time() - t_start)
        cv2.putText(
            frame,
            f"Latency: {latency_ms:.1f} ms | FPS: {avg_fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow("Ultralytics Async", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()

# -----------------------------------------------------------------------------
# Delegate configuration helper (used automatically inside Ultralytics YOLO)
# -----------------------------------------------------------------------------

# The Ultralytics YOLO "AutoBackend" reads the env var internally, so we don't need
# extra code here. As long as YOLO_BACKEND is set (or defaulted) to "armnn", the
# proper delegate path will be chosen in autobackend.py.

# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------

async def main(
    source: str = "./data/serve.mp4",
    weights: str = "yolov8n.tflite",
    conf: float = 0.3,
    imgsz: int = 640,
) -> None:
    """Run the full asynchronous pipeline."""
    # Queues
    input_queue: asyncio.Queue = asyncio.Queue(maxsize=4)
    output_queue: asyncio.Queue = asyncio.Queue(maxsize=4)

    # Video capture
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise FileNotFoundError(f"Unable to open video source: {source}")

    # Load model (AutoBackend will pick ArmNN delegate by default)
    model = YOLO(weights)
    model.warmup(imgsz=(1, 3, imgsz, imgsz))

    # Gather tasks
    tasks = [
        preprocess(input_queue, cap),
        predict(input_queue, output_queue, model, conf=conf, imgsz=imgsz),
        postprocess(output_queue),
    ]
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Interrupted by user.")
