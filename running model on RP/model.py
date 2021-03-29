import numpy as np
import tensorflow as tf

import pandas as pd

np.set_printoptions(suppress=True,
   formatter={'float_kind':'{:16.3f}'.format}, linewidth=130)


# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test the model on random input data.
input_shape = input_details[0]['shape']
#print("inputshape",input_shape)
#input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)

#Test the model on specific packets manually
#Impersonating
#input_data = np.array([1481193271.0,1088.0,0.0,8.0,255.0,240.0,0.0,0.0,255.0,8.0,9.0,0.0], dtype=np.float32)

#DOS
#input_data = np.array([0.0,0.0,8.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0], dtype=np.float32)

#Bengin
#input_data = np.array([0.000271,128.0,0.0,8.0,0.0,23.0,220.0,9.0,22.0,17.0,22.0,187.0], dtype=np.float32)

#fUZZY
input_data = np.array([259.050627,339.0,0.0,8.0,0.0,161.0,32.0,255.0,0.0,255.0,48.0,239.0], dtype=np.float32)




input_data = input_data.reshape(1,12,1)


interpreter.set_tensor(input_details[0]['index'], input_data)


print("inputdata",input_data)

interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
output_data = interpreter.get_tensor(output_details[0]['index'])


# for x in range (len(output_data)):
#     for i in range (len(output_data[x])):
#         #print('{:.2%}'.output_data[x][i])


maxindex = output_data.argmax()



if maxindex == 0:
    print("Bnign attack")
  
elif maxindex == 1:
    print("DOS Packet")

elif maxindex == 2:
    print("FUZZY attack")

elif maxindex == 3:
    print("IMPERSONATING attack")   



#print(output_data[0][1])

#print(type(output_data))

print(output_data)


