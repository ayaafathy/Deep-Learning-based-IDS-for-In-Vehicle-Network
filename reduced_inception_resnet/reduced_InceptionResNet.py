import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Model
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import Activation
from keras.layers import Dropout
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import concatenate


#print(tf.__version__)



#def buildDataFrame():




'''Defining a Convolution block that will be used throughout the network.'''
def conv_block(x, nb_filter, nb_row, nb_col, padding, strides, use_bias=False):

  x = Conv2D(nb_filter, (nb_row, nb_col), strides=strides, padding=padding, use_bias=use_bias)(x)
  x = BatchNormalization(axis=-1, momentum=0.9997, scale=False)(x)         #CHEEECKKK
  x = Activation('relu')(x)

  return x



'''
   Stem of the reduced Inception-ResNet. 
   This is the input part of the network 
   and produces feature maps from the input.
   -Input: 29×29×1
   -Output: 13×13×128 
'''
def stem(input):
  x = conv_block(input, 32, 3, 3, padding='same', strides=(1,1))
  x = conv_block(x, 32, 3, 3, padding='valid', strides=(1,1))
  x = MaxPooling2D((3,3), strides=(2,2), padding='valid')(x)
  x = conv_block(x, 64, 1, 1, padding='same', strides=(1,1))
  x = conv_block(x, 128, 1, 1, padding='same', strides=(1,1))
  x = conv_block(x, 128, 3, 3, padding='same', strides=(1,1))

  return x



'''
   Schema for a 13×13 grid (Inception-resnet-A) module of the reduced Inception-ResNet. 
   -Input & Output: 6×6×448
'''
def inception_resnet_A(input):
  a =  conv_block(input, 32, 1, 1, padding='same', strides=(1,1))

  b = conv_block(input, 32, 1, 1, padding='same', strides=(1,1))
  b = conv_block(b, 32, 3, 3, padding='same', strides=(1,1))

  c = conv_block(input, 32, 1, 1, padding='same', strides=(1,1))
  c = conv_block(c, 32, 3, 3, padding='same', strides=(1, 1))
  c = conv_block(c, 32, 3, 3, padding='same', strides=(1,1))

  abc = concatenate([a, b, c], axis= -1)
  #axis??????????

  x = conv_block(abc, 128, 1, 1, padding='same', strides=(1,1))
  #linear??? check paper

  return  x



'''
   Schema for a 6×6 grid (Inception-resnet-B) module of the reduced Inception-ResNet. 
   -Input & Output: 6×6×448
'''
def inception_resnet_B(input):
  a = conv_block(input, 64, 1, 1, padding='same', strides=(1, 1))

  b = conv_block(input, 64, 1, 1, padding='same', strides=(1, 1))
  b = conv_block(b, 64, 1, 3, padding='same', strides=(1, 1))
  b = conv_block(b, 64, 3, 1, padding='same', strides=(1, 1))

  ab = concatenate([a, b], axis=-1)
  # axis??????????

  x = conv_block(ab, 448, 1, 1, padding='same', strides=(1, 1))
  #linear??? check paper

  return x



'''
   Schema for a 13×13 grid-reduction (Reduction-A) module of the reduced Inception-ResNet. 
   -Input: 13×13×128
   -Output: 6×6×448 
'''
def reduction_A(input):
  a = MaxPooling2D((3,3), strides=(2,2), padding='valid')(input)

  b = conv_block(input, 192, 3, 3, padding='valid', strides=(2,2))

  c = conv_block(input, 96, 1, 1, padding='same', strides=(1,1))
  c = conv_block(c, 96, 3, 3, padding='same', strides=(1,1))
  c = conv_block(c, 128, 3, 3, padding='valid', strides=(2,2))

  x = concatenate([a, b, c], axis= -1)

  return x




'''
   Schema for a 6×6 grid-reduction (Reduction-B) module of the reduced Inception-ResNet. 
   -Input: 6×6×448
   -Output: 2×2×896
'''
def reduction_B(input):
  a = MaxPooling2D((3,3), strides=(1,1), padding='valid')(input)

  b = conv_block(input, 128, 1, 1, padding='same', strides=(1,1))
  b = conv_block(b, 192, 3, 3, padding='valid', strides=(1,1))

  c = conv_block(input, 128, 1, 1, padding='same', strides=(1,1))
  c = conv_block(c, 128, 3, 3, padding='valid', strides=(1,1))

  d = conv_block(input, 128, 1, 1, padding='same', strides=(1,1))
  d = conv_block(d, 128, 3, 3, padding='same', strides=(1,1))
  d = conv_block(d, 128, 3, 3, padding='valid', strides=(1, 1))

  x = concatenate([a, b, c, d], axis=-1)

  return  x







def build_model():

  input = Input((29, 29, 1))        #CHECKKK THISS AFTER STRUCTURING THE DATA
  '''Reduced Inception ResNet building blocks'''
  x = stem(input)
  x = inception_resnet_A(x)
  x = reduction_A(x)
  x = inception_resnet_B(x)
  x = reduction_B(x)
  x = AveragePooling2D((2,2))(x)   #CHECK DIMENSIONS
  x = Dropout(0.5)(x)              #CHECK PERCENTAGE

  Model()
  model = Model(input, x, name='Reduced Inception ResNet')

  #model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

  '''Adding SoftMax layer'''
  reduced_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

  return reduced_model






def main():
  reduced_model = build_model()
  reduced_model.summary()
  #reduced_model.fit()
