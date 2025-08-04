import numpy as np
import cv2

def convert_to_binary(input_data, binary_path):
    input_data.astype(np.float32).tofile(binary_path)
    return binary_path
    
def convert_to_numpy(output_handlers_with_shape, dtype = np.float32):
    outputs = []
    for output_handler in output_handlers_with_shape:
        data = np.fromfile(output_handler, dtype=dtype)
        outputs.append(np.reshape(data, output_handlers_with_shape[output_handler]))
    return np.array(outputs)[0]
