import argparse
import time
import warnings
import os
import shutil
import numpy as np
import tensorflow as tf
import sys

# 假設 neuronrt module 路徑正確且在 PYTHONPATH
from utils.neuronpilot.data import convert_to_binary, convert_to_numpy  # 修正拼字

# NeuronRT import deferred for clearer error message
warnings.simplefilter("ignore")

parser = argparse.ArgumentParser(description="Benchmark TFLite model on NeuronRT device (e.g., MDLA/VPU).")
parser.add_argument("-m", "--tflite_model", type=str, required=True, help="Path to .tflite model file.")
parser.add_argument(
    "-d",
    "--device",
    type=str,
    default="mdla3.0",
    choices=["mdla3.0", "mdla2.0", "vpu"],
    help="NeuronRT device to use (e.g., mdla3.0, mdla2.0, vpu).",
)
parser.add_argument("-t", "--iteration", default=10, type=int, help="Number of timed inference iterations.")
args = parser.parse_args()

# Load NeuronRT interpreter
try:
    from utils.neuronpilot import neuronrt
except ImportError as e:
    print(f"[ERROR] Failed to import neuronrt: {e}", file=sys.stderr)
    sys.exit(1)

try:
    interpreter = neuronrt.Interpreter(model_path=args.tflite_model, device=args.device)
except Exception as e:
    print(f"[ERROR] Failed to create NeuronRT Interpreter: {e}", file=sys.stderr)
    sys.exit(1)

# Allocate tensors
try:
    interpreter.allocate_tensors()
except Exception as e:
    print(f"[ERROR] allocate_tensors() failed: {e}", file=sys.stderr)
    sys.exit(1)

# Get I/O metadata
try:
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
except Exception as e:
    print(f"[ERROR] Failed to get I/O details: {e}", file=sys.stderr)
    sys.exit(1)

# Prepare random input matching shape and dtype
input_shape = input_details[0]["shape"]
input_dtype = input_details[0]["dtype"]
try:
    inputs = np.random.rand(*input_shape).astype(input_dtype)
except Exception as e:
    print(f"[ERROR] Failed to create input tensor: {e}", file=sys.stderr)
    sys.exit(1)

# Set tensor (measure)
t_set_start = time.time()
try:
    interpreter.set_tensor(input_details[0]["index"], inputs)
except Exception as e:
    print(f"[ERROR] set_tensor failed: {e}", file=sys.stderr)
    sys.exit(1)
t_set_end = time.time()

# Warmup (exclude from timing)
try:
    interpreter.invoke()
except Exception as e:
    print(f"[WARN] Warmup invoke failed: {e}", file=sys.stderr)

# Timed inference loop
times_ms = []
for i in range(args.iteration):
    try:
        t0 = time.time()
        interpreter.invoke()
        t1 = time.time()
        times_ms.append((t1 - t0) * 1000.0)
    except Exception as e:
        print(f"[WARN] invoke failed on iteration {i}: {e}", file=sys.stderr)

# Get output (measure)
t_get_start = time.time()
try:
    outputs = interpreter.get_tensor(output_details[0]["index"])
except Exception as e:
    print(f"[ERROR] get_tensor failed: {e}", file=sys.stderr)
    sys.exit(1)
t_get_end = time.time()

# Report
print("=== NeuronRT Benchmark ===")
print(f"Model              : {args.tflite_model}")
print(f"Device             : {args.device}")
print(f"Iteration count    : {args.iteration}")
print(f"Input shape        : {input_shape}, dtype={input_dtype}")
print(f"Set tensor time    : {(t_set_end - t_set_start) * 1000:.3f} ms")
if times_ms:
    avg_inf = float(np.mean(times_ms))
    std_inf = float(np.std(times_ms))
    print(f"Inference avg      : {avg_inf:.3f} ms over {len(times_ms)} runs")
    print(f"Inference stddev   : {std_inf:.3f} ms")
else:
    print("[WARN] No successful inference iterations recorded.")
print(f"Get tensor time    : {(t_get_end - t_get_start) * 1000:.3f} ms")
print(f"Output shape       : {outputs.shape if hasattr(outputs, 'shape') else 'unknown'}")

# Sanity check on output
if isinstance(outputs, np.ndarray):
    if not np.all(np.isfinite(outputs)):
        print("[WARN] Output contains non-finite values.", file=sys.stderr)
else:
    print("[WARN] Output is not a numpy array; cannot sanity-check.", file=sys.stderr)

