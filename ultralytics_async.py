'''
ultralytics_async.py (GPUAcc Default)
====================================
Asynchronous video inference pipeline for Ultralytics YOLO **with ArmNN ‑ GpuAcc 作為預設後端**。

* 若使用者「沒有」設定任何環境變數，程式會自動：
  - `YOLO_BACKEND = "armnn"`
  - `YOLO_ARMNN_BACKEND = "GpuAcc"`
* 如需改成 CpuAcc 或 NeuronRT，只要在 `import YOLO` 之前覆寫相同環境變數即可。
'''

import os
import time
import asyncio
from typing import Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

# -----------------------------------------------------------------------------
# 全域環境變數：強制預設 ArmNN + GpuAcc
# -----------------------------------------------------------------------------
# setdefault 只在外部沒預先 export 時才生效
os.environ.setdefault("YOLO_BACKEND", "armnn")
os.environ.setdefault("YOLO_ARMNN_BACKEND", "GpuAcc")

# 讀取最終環境值（可能被使用者覆寫）
backend_choice  = os.getenv("YOLO_BACKEND").lower()
armnn_backend   = os.getenv("YOLO_ARMNN_BACKEND")
neuron_device   = os.getenv("YOLO_NEURON_DEVICE", "mdla3.0")

print(
    f"[Backend] YOLO_BACKEND={backend_choice} | "
    f"YOLO_ARMNN_BACKEND={armnn_backend} | "
    f"YOLO_NEURON_DEVICE={neuron_device}"
)

# -----------------------------------------------------------------------------
# Async pipeline helpers
# -----------------------------------------------------------------------------

async def preprocess(input_q: asyncio.Queue, cap: cv2.VideoCapture) -> None:
    """Capture frames and push with timestamp."""
    loop = asyncio.get_running_loop()
    while cap.isOpened():
        ret, frame = await loop.run_in_executor(None, cap.read)
        if not ret:
            await input_q.put(None)
            break
        await input_q.put((frame, time.time()))
    cap.release()
    await input_q.put(None)

async def predict(
    input_q: asyncio.Queue,
    output_q: asyncio.Queue,
    model: YOLO,
    conf: float = 0.3,
    imgsz: int = 640,
) -> None:
    """Run inference in executor thread."""
    loop = asyncio.get_running_loop()
    while True:
        item = await input_q.get()
        if item is None:
            await output_q.put(None)
            break
        frame, ts = item
        result = await loop.run_in_executor(
            None,
            lambda: model.predict(frame, conf=conf, imgsz=imgsz, stream=False)[0].plot(),
        )
        await output_q.put((result, ts))

async def postprocess(output_q: asyncio.Queue) -> None:
    """Display frames + stats."""
    t0 = time.time(); n = 0
    while True:
        item = await output_q.get()
        if item is None:
            break
        frame, ts = item
        latency = (time.time() - ts) * 1e3
        n += 1; fps = n / (time.time() - t0)
        cv2.putText(frame, f"Latency {latency:.1f} ms | FPS {fps:.1f}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2, cv2.LINE_AA)
        cv2.imshow("Ultralytics Async", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------

async def main(
    source: str = "./data/serve.mp4",
    weights: str = "./models/yolov8n_float32.tflite",
    conf: float = 0.3,
    imgsz: int = 640,
) -> None:
    """Launch end‑to‑end async pipeline."""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise FileNotFoundError(f"Unable to open video source: {source}")

    input_q: asyncio.Queue  = asyncio.Queue(maxsize=4)
    output_q: asyncio.Queue = asyncio.Queue(maxsize=4)

    model = YOLO(weights)  # AutoBackend 會依環境變數載入 ArmNN GpuAcc

    # Warm‑up (1 dummy frame)
    try:
        _ = model.predict(np.zeros((imgsz, imgsz, 3), dtype=np.uint8), verbose=False)
        print("[INFO] Warm‑up done")
    except Exception as e:
        print(f"[WARN] Warm‑up failed: {e}")

    tasks = [
        preprocess(input_q, cap),
        predict(input_q, output_q, model, conf=conf, imgsz=imgsz),
        postprocess(output_q),
    ]
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Interrupted by user.")
