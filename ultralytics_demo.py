from ultralytics import YOLO
import cv2, asyncio, time, argparse

async def preprocess(input_queue, cap):
    loop = asyncio.get_running_loop()
    while cap.isOpened():
        ret, frame = await loop.run_in_executor(None, cap.read)
        if not ret:
            await input_queue.put(None)
        else:
            await input_queue.put(frame)
    cap.release()

async def predict(input_queue, output_queue, model):
    loop = asyncio.get_running_loop()
    while True:
        frame = await input_queue.get()
        if frame is not None:  
            result = await loop.run_in_executor(
                None,
                lambda: model.predict(frame, stream=False)[0],
            )
            await output_queue.put(result)

async def postprocess(output_queue):
    while True:
        t = time.time()
        result = await output_queue.get()
        if result is not None:
            result = result.plot()
            fps = 1/ (time.time() - t)
            cv2.putText(result , f"FPS {fps:.1f}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 5, cv2.LINE_AA
            )
            cv2.imshow("Ultralytics Async", cv2.resize(result , (480, 360)))
            cv2.waitKey(1)
    cv2.destroyAllWindows()

async def main():
    parser = argparse.ArgumentParser(description='Ultralytics YOLO 非同步串流推論')
    parser.add_argument('--tflite_model', type=str, default='./models/yolov8n_float32.tflite', help='')
    parser.add_argument('--video_path', type=str, default='./data/video.mp4', help='影片檔案路徑')
    args = parser.parse_args()
    
    input_queue = asyncio.Queue()
    output_queue = asyncio.Queue()
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"{args.video_path} not found.")
    model = YOLO(args.tflite_model, task='detect')
    
    print(f"開始處理影片: {args.video_path}")
    await asyncio.gather(
        preprocess(input_queue, cap),
        predict(input_queue, output_queue, model),
        postprocess(output_queue)
    )

if __name__ == "__main__":
    asyncio.run(main())




