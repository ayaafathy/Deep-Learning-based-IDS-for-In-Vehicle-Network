#!/usr/bin/env python
# coding: utf-8

# In[10]:


#%%
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
from keras.models import load_model
from keras import optimizers


# Load training data
train = pd.read_csv(r"E:\University\GP\LSTM\Datasets\file.csv")
train.head(10000000)


# In[7]:





# In[ ]:


# Scale features
s1 = MinMaxScaler(feature_range=(-1,1))
Xs = s1.fit_transform(train[['Time','IntID1', 'IntID2', 'DLC', 'IntData1','IntData2','IntData3','IntData4','IntData5','IntData6', 'IntData7','IntData8', 'Label']])

# Scale predicted value
s2 = MinMaxScaler(feature_range=(-1,1))
Ys = s2.fit_transform(train[['Label']])
# Each time step uses last 'window' to predict the next change
window = 25
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
model.add(LSTM(512, return_sequences=True,           input_shape=(X.shape[1],X.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(8, return_sequences=True))
model.add(Dropout(0.2))
model.add(Dense(units=1, activation='relu'))
opt = optimizers.Adamax(lr=0.001)
model.compile(optimizer = opt , loss = 'binary_crossentropy',              metrics = ['accuracy'])

# Allow for early exit
es = EarlyStopping(monitor='loss',mode='min',verbose=1,patience=10)

# Fit (and time) LSTM model
t0 = time.time()
history = model.fit(X, Y, epochs = 10, batch_size = 512, callbacks=[es], verbose=1)
t1 = time.time()
print('Runtime: %.2f s' %(t1-t0))

# Plot loss
plt.figure(figsize=(8,4))
plt.semilogy(history.history['loss'])
plt.xlabel('epoch'); plt.ylabel('loss')
plt.savefig('tclab_loss.png')
model.save('model.h5')

# Verify the fit of the model
Yp = model.predict(X)

# un-scale outputs
Yu = s2.inverse_transform(Yp)
Ym = s2.inverse_transform(Y)


# In[ ]:





# In[ ]:




