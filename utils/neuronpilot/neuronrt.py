from utils.neuronpilot.data import convert_to_binary, conert_to_numpy
from utils.tools import Neuronpilot_WebAPI, tflite_to_dla
import numpy as np
import tensorflow as tf
import os, shutil, subprocess

class Interpreter():
    def __init__(self, model_path, device = 'mdla3.0'):
        self.tflite_path = model_path
        self.device = device
        interpreter = tf.lite.Interpreter(model_path=model_path)
        self.input_details = interpreter.get_input_details()
        self.output_details = interpreter.get_output_details()
        
    def allocate_tensors(self):
        self.seed = int(np.random.rand() * 10e12)
        self.dla_path = tflite_to_dla(self.tflite_path, self.device)

        if not os.path.exists(self.dla_path):
           self.dla_path = Neuronpilot_WebAPI(
               tflite_path = self.tflite_path, output_folder = './models', 
               device = 'mdla3.0',
               url = 'https://app-aihub-neuronpilot.azurewebsites.net/'
           )
           print(f"Converted file saved to: {self.dla_path}")
        else:
           print(f"Load existed dla model: {self.dla_path}")
        load(self.dla_path)
        
        if os.path.exists(f'./bin/{self.seed}'):
           shutil.rmtree(f'./bin/{self.seed}')
        os.mkdir(f'./bin/{self.seed}') 
        self.input_handlers = [f'./bin/{self.seed}/input{i}.bin' for i, input in enumerate(self.input_details)]
        self.output_handlers_with_shape = {f'./bin/{self.seed}/output{i}.bin': tuple([1] + output['shape'].tolist()) for i, output in enumerate(self.output_details)}

    def get_input_details(self):
        return self.input_details

    def get_output_details(self):
        return self.output_details
   
    def set_tensor(self, _, inputs):
        for input, binary_path in zip(inputs, self.input_handlers):
            input_data = np.array([input]).astype(self.input_details[0]['dtype'])
            self.bin_path = convert_to_binary(input_data, binary_path)

    def invoke(self):
        predict(input_handlers = self.input_handlers, 
            output_handlers  = list(self.output_handlers_with_shape.keys()), 
            compiled_dla_model = self.dla_path, 
        )
        
    def get_tensor(self, _):
        return conert_to_numpy(self.output_handlers_with_shape, dtype = np.float32)[0]


def load(compiled_dla_model):
    commands = ["sudo", "neuronrt", "-a", compiled_dla_model, "-d"]
    results = subprocess.run(commands, capture_output=True, text=True)
    
def predict(input_handlers, output_handlers, compiled_dla_model, boost_value = 100):
    commands = ["sudo", "neuronrt",  
                "-m",  "hw",  
                "-a",  compiled_dla_model,
                "-c",  "1",         # Repeat the inference <num> times. It can be used for profiling.
                "-b",  str(boost_value),     
                "-i",  ' -i '.join(input_handlers),  
                "-o",  ' -o '.join(output_handlers)]
    p = subprocess.Popen(commands, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    p.wait()
