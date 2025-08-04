import os
import cv2
import time
import asyncio
import numpy as np
from ultralytics import YOLO

# --- Configuration / Backend selection ---
# Set backend via environment variable: "armnn", "neuronrt", or leave unset for default TFLite/EdgeTPU
backend_choice = os.getenv("YOLO_BACKEND")  # e.g., export YOLO_BACKEND=armnn
# Optionally set specific devices if needed via env too
neuron_device = os.getenv("YOLO_NEURON_DEVICE", "mdla3.0")  # used if neuronrt backend
armnn_backend = os.getenv("YOLO_ARMNN_BACKEND", "GpuAcc")  # used if armnn backend

# --- Async pipeline ---
async def preprocess(input_queue, cap):
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            await input_queue.put(None)
            break
        timestamp = time.time()
        await input_queue.put((frame, timestamp))
    cap.release()
    await input_queue.put(None)  # ensure downstream termination

async def predict(input_queue, output_queue, model):
    loop = asyncio.get_running_loop()
    while True:
        item = await input_queue.get()
        if item is None:
            await output_queue.put(None)
            break
        frame, t0 = item
        try:
            # Run blocking predict in threadpool to avoid blocking event loop
            results = await loop.run_in_executor(None, lambda: model.predict(frame, verbose=False))
            plotted = results[0].plot()
            await output_queue.put((plotted, t0))
        except Exception as e:
            print(f"[ERROR] model.predict failed: {e}")
            await output_queue.put(None)
            break

async def postprocess(output_queue):
    # Simple FPS smoothing
    frame_count = 0
    start_time_total = time.time()
    while True:
        item = await output_queue.get()
        if item is None:
            break
        result, t0 = item
        if result is None:
            continue
        frame_count += 1
        end_time = time.time()
        latency_ms = (end_time - t0) * 1000.0
        elapsed = end_time - start_time_total
        fps = frame_count / elapsed if elapsed > 0 else 0.0
        cv2.imshow("streaming", result)
        print(f"[INFO] End-to-end latency: {latency_ms:.1f} ms | Avg FPS: {fps:.2f}")
        if cv2.waitKey(1) == ord("q"):
            break
    cv2.destroyAllWindows()

async def main():
    input_queue = asyncio.Queue(maxsize=4)
    output_queue = asyncio.Queue(maxsize=4)

    cap = cv2.VideoCapture("./data/serve.mp4")
    if not cap.isOpened():
        print("[ERROR] Failed to open video source.")
        return

    # Load model; the underlying AutoBackend should pick up YOLO_BACKEND env var you set
    model_path = "./models/yolov8n_float32.tflite"
    model = YOLO(model_path, task="detect")

    # Warmup (single dummy inference to reduce first-run overhead)
    try:
        _ = await asyncio.get_running_loop().run_in_executor(None, lambda: model.predict(np.zeros((640, 640, 3), dtype="uint8"), verbose=False))
        print("[INFO] Warmup done.")
    except Exception as e:
        print(f"[WARN] Warmup failed: {e}")

    await asyncio.gather(
        preprocess(input_queue, cap),
        predict(input_queue, output_queue, model),
        postprocess(output_queue),
    )


