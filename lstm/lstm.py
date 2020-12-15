import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Model
from keras.models import Sequential
from keras.layers import Input
from keras.layers import Activation
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers.merge import concatenate



epochs_num = 500
batch_size = 512
learning_rate = 0.001
hidden_layer_activation = tf.keras.activations.relu
used_optimizer = tf.keras.optimizers.Adamax(learning_rate, name='Adamax')
loss_function = tf.keras.losses.binary_crossentropy
#Activation In hidden layers: RelU
#Activation in Output: Sigmoid
#Loss Function: Binary cross entropy (BCE)
#Neurons: 512 in LSTM, 8 in FC, 1 in output


def lstm_model():
  lstm = Sequential()
  lstm.add(LSTM(512, activation='RelU'))
  lstm.add(Dense(units = 8))
