import argparse, time, warnings
import numpy as np
import tensorflow as tf

warnings.simplefilter('ignore')

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--tflite_model", type=str, help="Path to .tflite")
parser.add_argument("-d", "--device", default='GpuAcc', choices=['CpuAcc', 'GpuAcc'], type=str, help="Device name for acceleration")
parser.add_argument("-t", "--iteration", default=10, type=int, help="Test How Many Times?")
args = parser.parse_args()

# Initialize Interpreter
Interpreter, load_delegate = tf.lite.Interpreter, tf.lite.experimental.load_delegate
armnn_delegate = load_delegate( library="/home/ubuntu/armnn/ArmNN-linux-aarch64/libarmnnDelegate.so",
                                       options={"backends": args.device, "logging-severity":"info"})
interpreter = Interpreter(model_path=args.tflite_model, 
                          experimental_delegates=[armnn_delegate])
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Forward Propagation
inputs = np.random.rand(*input_details[0]['shape']).astype(input_details[0]['dtype'])

t1 = time.time()
interpreter.set_tensor(input_details[0]['index'], inputs)
t2 = time.time()
for _ in range(args.iteration):
  interpreter.invoke()
  
t3 = time.time()
outputs = interpreter.get_tensor(output_details[0]['index'])

print(f'Set tensor speed: {(t2 - t1) * 1000} ms')
print(f'Inference speed: {(t3 - t2) * 1000 / args.iteration} ms')
print(f'Get tensor speed: {(time.time() - t3) * 1000} ms')
