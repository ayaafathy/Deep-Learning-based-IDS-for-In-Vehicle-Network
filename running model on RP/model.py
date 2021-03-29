import numpy as np
import tensorflow as tf

import pandas as pd



# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test the model on random input data.
input_shape = input_details[0]['shape']
#print("inputshape",input_shape)
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)

#Test the model on specific packets manually
input_data = np.array([1481193271.0,1088.0,0.0,8.0,255.0,240.0,0.0,0.0,255.0,8.0,9.0,0.0], dtype=np.float32)

#input_data = np.array([0.0,0.0,8.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0], dtype=np.float32)
#input_data = pd.read_csv("FinalDataset2.csv", usecols = ['Time','IntID1','IntID2','IntID3','IntID4','IntID5','IntID6','IntID7','IntID8')


#input_data = np.array(np.all(input_shape), dtype =np.float32)
input_data = input_data.reshape(1,12,1)


interpreter.set_tensor(input_details[0]['index'], input_data)


print("inputdata",input_data)

interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)