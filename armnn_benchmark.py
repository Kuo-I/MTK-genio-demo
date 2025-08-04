import argparse
import time
import warnings
import numpy as np
import tensorflow as tf
import sys

warnings.simplefilter("ignore")

# ---- CLI ----
parser = argparse.ArgumentParser(
    description="Run TFLite/ArmNN/NeuronRT inference benchmark for a YOLO-derived model."
)
parser.add_argument(
    "-m",
    "--tflite_model",
    type=str,
    required=True,
    help="Path to the .tflite model file.",
)
parser.add_argument(
    "-b",
    "--backend",
    choices=["tflite", "armnn", "neuronrt"],
    default="armnn",
    help="Which backend to use for inference.",
)
parser.add_argument(
    "-d",
    "--device",
    type=str,
    default="GpuAcc",
    help="Backend-specific device: for armnn use 'CpuAcc' or 'GpuAcc'; for neuronrt use e.g. 'mdla3.0'.",
)
parser.add_argument(
    "-t", "--iteration", default=10, type=int, help="Number of timed inference iterations."
)
args = parser.parse_args()


def load_armnn_interpreter(model_path: str, armnn_backend: str):
    Interpreter, load_delegate = tf.lite.Interpreter, tf.lite.experimental.load_delegate
    try:
        armnn_delegate = load_delegate(
            library="/home/ubuntu/armnn/ArmNN-linux-aarch64/libarmnnDelegate.so",
            options={"backends": armnn_backend, "logging-severity": "info"},
        )
    except Exception as e:
        print(f"[WARN] Failed to load ArmNN delegate: {e}. Falling back to pure TFLite.", file=sys.stderr)
        return tf.lite.Interpreter(model_path=model_path)
    interpreter = Interpreter(model_path=model_path, experimental_delegates=[armnn_delegate])
    return interpreter


def load_neuronrt_interpreter(model_path: str, neuron_device: str):
    try:
        from utils.neuronpilot import neuronrt
    except ImportError as e:
        raise RuntimeError(f"Cannot import neuronrt module: {e}")
    # Assumes NeuronRT's API is similar to TFLite-style
    interpreter = neuronrt.Interpreter(model_path=model_path, device=neuron_device)
    return interpreter


def load_tflite_interpreter(model_path: str, use_gpu: bool = False):
    # Standard TFLite; GPU delegate could be added here if desired
    Interpreter = tf.lite.Interpreter
    return Interpreter(model_path=model_path)


# ---- Initialization ----
if args.backend == "armnn":
    interpreter = load_armnn_interpreter(args.tflite_model, args.device)
elif args.backend == "neuronrt":
    try:
        interpreter = load_neuronrt_interpreter(args.tflite_model, args.device)
    except Exception as e:
        print(f"[ERROR] NeuronRT backend failed: {e}. Exiting.", file=sys.stderr)
        sys.exit(1)
else:  # tflite
    interpreter = load_tflite_interpreter(args.tflite_model)

# Allocate / prepare
try:
    interpreter.allocate_tensors()
except Exception as e:
    print(f"[ERROR] allocate_tensors() failed: {e}", file=sys.stderr)
    sys.exit(1)

# Get I/O metadata
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Prepare random input
input_shape = input_details[0]["shape"]
input_dtype = input_details[0]["dtype"]
inputs = np.random.rand(*input_shape).astype(input_dtype)

# Set tensor (measure separately)
t_set_start = time.time()
interpreter.set_tensor(input_details[0]["index"], inputs)
t_set_end = time.time()

# Warmup
try:
    interpreter.invoke()
except Exception as e:
    print(f"[WARN] Warmup invoke failed: {e}", file=sys.stderr)

# Timed inference
times = []
for i in range(args.iteration):
    # Optionally re-set input each iteration if desired; here we keep same input
    t0 = time.time()
    interpreter.invoke()
    t1 = time.time()
    times.append((t1 - t0) * 1000)  # ms

# Get output (measure)
t_get_start = time.time()
outputs = interpreter.get_tensor(output_details[0]["index"])
t_get_end = time.time()

# Summary
avg_inf = sum(times) / len(times)
print("=== Benchmark Result ===")
print(f"Backend           : {args.backend}")
print(f"Model             : {args.tflite_model}")
print(f"Input shape       : {input_shape}, dtype={input_dtype}")
print(f"Set tensor time   : {(t_set_end - t_set_start) * 1000:.3f} ms")
print(f"Inference avg     : {avg_inf:.3f} ms over {args.iteration} runs")
print(f"Inference stddev  : {np.std(times):.3f} ms")
print(f"Get tensor time   : {(t_get_end - t_get_start) * 1000:.3f} ms")
print(f"Output shape      : {outputs.shape if hasattr(outputs, 'shape') else 'unknown'}")

# Simple sanity check (e.g., finite)
if isinstance(outputs, np.ndarray):
    if not np.all(np.isfinite(outputs)):
        print("[WARN] Output contains non-finite values.", file=sys.stderr)
else:
    print("[WARN] Output is not numpy array; cannot sanity-check.", file=sys.stderr)
