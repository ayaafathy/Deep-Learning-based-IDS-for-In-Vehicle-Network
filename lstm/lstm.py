#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import time
# For LSTM model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.callbacks import EarlyStopping
from keras import optimizers


# Load dataset
train = pd.read_csv(r"C:\Users\Mostafa\PycharmProjects\Deep-Learning-based-IDS-for-In-Vehicle-Network\Datasets\file.csv")
# train.head(100000)
# train.describe()


# # Scale features
# s1 = MinMaxScaler(feature_range=(-1,1))
# Xs = s1.fit_transform(train[['Time','IntID1', 'IntID2', 'DLC', 'IntData1','IntData2','IntData3','IntData4','IntData5','IntData6', 'IntData7','IntData8', 'Label']])
#
# # Scale predicted value
# s2 = MinMaxScaler(feature_range=(0,1))
# Ys = s2.fit_transform(train[['Label']])
# # Each time step uses last 'window' to predict the next change
# window = 50
# X = []
# Y = []
# for i in range(window,len(Xs)):
#     X.append(Xs[i-window:i,:])
#     Y.append(Ys[i])
#
# # Reshape data to format accepted by LSTM
# X, Y = np.array(X), np.array(Y)
#
# # create and train LSTM model
#
# # Initialize LSTM model
# model = Sequential()
#
# model.add(LSTM(512, return_sequences=True,  activation='relu', \
#           input_shape=(X.shape[1],X.shape[2])))
# model.add(Dropout(0.2))
# model.add(Dense(8, activation='relu'))
# model.add(Dropout(0.2))
# model.add(LSTM(units=1, activation='relu'))
# opt = optimizers.Adamax(lr=0.001)
# model.compile(optimizer = opt , loss = 'binary_crossentropy',\
#               metrics = ['accuracy'])
#
# # Allow for early exit
# es = EarlyStopping(monitor='loss',mode='min',verbose=1,patience=10)
#
# # Fit (and time) LSTM model
# t0 = time.time()
# history = model.fit(X, Y, epochs = 10, batch_size = 512, validation_split=0.4, callbacks=[es], verbose=1)
# t1 = time.time()
# print('Runtime: %.2f s' %(t1-t0))
#
# # Plot loss
# plt.figure(figsize=(8,4))
# plt.semilogy(history.history['loss'])
# plt.xlabel('epoch'); plt.ylabel('loss')
# plt.savefig('tclab_loss.png')
# model.save('model.h5')
#
# # Verify the fit of the model
# Yp = model.predict(X, verbose=1)
# print(Yp)
#
#

# In[6]:


# Scale features
s1 = MinMaxScaler(feature_range=(-1,1))
Xs = s1.fit_transform(train[['IntID1', 'IntID2', 'DLC', 'IntData1','IntData2','IntData3','IntData4','IntData5','IntData6', 'IntData7','IntData8', 'Label']])

# Scale predicted value
s2 = MinMaxScaler(feature_range=(0,1))
Ys = s2.fit_transform(train[['Label']])
# Each time step uses last 'window' to predict the next change
window = 1
X = []
Y = []
for i in range(window,len(Xs)):
    X.append(Xs[i-window:i,:])
    Y.append(Ys[i])

# Reshape data to format accepted by LSTM
X, Y = np.array(X), np.array(Y)

# create and train LSTM model

# Initialize LSTM model
model = Sequential()

model.add(LSTM(512, return_sequences=True,  activation='tanh', input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(8, activation='tanh'))
model.add(Dropout(0.2))
model.add(LSTM(units=1, activation='sigmoid'))
opt = optimizers.Adam(lr=0.0001)
model.compile(optimizer = opt , loss = 'binary_crossentropy', metrics = ['accuracy', 'Precision', 'Recall'])

# Allow for early exit
es = EarlyStopping(monitor='loss',mode='min',verbose=1,patience=10)

# Fit (and time) LSTM model
t0 = time.time()
history = model.fit(X, Y, epochs = 20, batch_size = 512, validation_split=0.2, callbacks=[es], verbose=1)
t1 = time.time()
print('Runtime: %.2f s' %(t1-t0))

# Plot loss
plt.figure(figsize=(8,4))
plt.semilogy(history.history['loss'])
plt.xlabel('epoch'); plt.ylabel('loss')
plt.savefig('loss.png')
model.save('model.h5')

# Verify the fit of the model
Yp = model.predict(X, verbose=1)



