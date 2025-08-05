''''
ultralytics_async.py (ArmNN‑GpuAcc default, selectable)
=====================================================
非同步影片推論範例，預設使用 **ArmNN + GpuAcc**；
但只要在執行前覆蓋 `YOLO_BACKEND` 即可切換至 **CpuAcc**、**NeuronRT**（MDLA 3.0）或其它自訂後端。

* 無任何環境變數時：
  ```
  YOLO_BACKEND        = armnn
  YOLO_ARMNN_BACKEND  = GpuAcc
  YOLO_NEURON_DEVICE  = mdla3.0  (僅 neuronrt 分支會用)
  ```
* 例如改跑 MDLA：
  ```bash
  YOLO_BACKEND=neuronrt python3 ultralytics_async.py
  ```
'''

import os
import time
import asyncio
import cv2
import numpy as np
from ultralytics import YOLO

# -----------------------------------------------------------------------------
# ❶ 預設環境變數（可被外部覆寫）
# -----------------------------------------------------------------------------
os.environ.setdefault("YOLO_BACKEND", "armnn")          # armnn / neuronrt / cpu / edgetpu ...
os.environ.setdefault("YOLO_ARMNN_BACKEND", "GpuAcc")   # GpuAcc / CpuAcc
os.environ.setdefault("YOLO_NEURON_DEVICE", "mdla3.0")   # mdla3.0 / vpu0  (NeuronRT 專用)

# ❷ 讀最終值並顯示
backend_choice = os.getenv("YOLO_BACKEND").lower()
armnn_backend  = os.getenv("YOLO_ARMNN_BACKEND")
neuron_device  = os.getenv("YOLO_NEURON_DEVICE")
# ---- 最終後端訊息 ----
if backend_choice == "neuronrt":
    print(f"[Backend] YOLO_BACKEND=neuronrt | YOLO_NEURON_DEVICE={neuron_device}")
else:
    print(f"[Backend] YOLO_BACKEND={backend_choice} | YOLO_ARMNN_BACKEND={armnn_backend}")

# -----------------------------------------------------------------------------
# Async helpers
# -----------------------------------------------------------------------------
async def preprocess(q_in: asyncio.Queue, cap: cv2.VideoCapture):
    loop = asyncio.get_running_loop()
    while cap.isOpened():
        ret, frame = await loop.run_in_executor(None, cap.read)
        if not ret:
            await q_in.put(None); break
        await q_in.put((frame, time.time()))
    cap.release(); await q_in.put(None)

async def predict(q_in: asyncio.Queue, q_out: asyncio.Queue, model: YOLO, conf=0.3, imgsz=640):
    loop = asyncio.get_running_loop()
    while True:
        item = await q_in.get()
        if item is None:
            await q_out.put(None); break
        frame, ts = item
        plotted = await loop.run_in_executor(
            None,
            lambda: model.predict(frame, conf=conf, imgsz=imgsz, stream=False)[0].plot(),
        )
        await q_out.put((plotted, ts))

async def postprocess(q_out: asyncio.Queue):
    n, t0 = 0, time.time()
    while True:
        item = await q_out.get()
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
# Main
# -----------------------------------------------------------------------------
async def main(source="./data/serve.mp4", weights="./models/yolov8n_float32.tflite", conf=0.3, imgsz=640):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise FileNotFoundError(f"Unable to open {source}")

    q_in, q_out = asyncio.Queue(4), asyncio.Queue(4)
    model = YOLO(weights)   # AutoBackend 依環境變數載入 delegate

    try:
        model.predict(np.zeros((imgsz, imgsz, 3), dtype=np.uint8), verbose=False)
        print("[INFO] Warm‑up done")
    except Exception as e:
        print(f"[WARN] Warm‑up failed: {e}")

    await asyncio.gather(
        preprocess(q_in, cap),
        predict(q_in, q_out, model, conf, imgsz),
        postprocess(q_out),
    )

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Interrupted by user.")
