#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import time
# For LSTM model
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.callbacks import EarlyStopping
from keras.models import load_model
from keras import optimizers



# Load training data
train = pd.read_csv(r"E:\University\GP\LSTM\Datasets\file.csv")
test = pd.read_csv(r"E:\University\GP\LSTM\Datasets\testfinal.csv")

# In[4]:


train.describe()


# In[12]:


# Scale features
s1 = MinMaxScaler(feature_range=(-1,1))
Xs = s1.fit_transform(train[['IntID1', 'IntID2', 'DLC', 'IntData1','IntData2','IntData3','IntData4','IntData5','IntData6', 'IntData7','IntData8', 'Label']])

# Scale predicted value
s2 = MinMaxScaler(feature_range=(0,1))
Ys = s2.fit_transform(train[['Label']])

# Scale features
t1 = MinMaxScaler(feature_range=(-1,1))
Xtest = s1.fit_transform(train[['IntID1', 'IntID2', 'DLC', 'IntData1','IntData2','IntData3','IntData4','IntData5','IntData6', 'IntData7','IntData8', 'Label']])

# Scale predicted value
t2 = MinMaxScaler(feature_range=(0,1))
Ytest = s2.fit_transform(train[['Label']])


# Each time step uses last 'window' to predict the next change
window = 1
X = []
Y = []
for i in range(window,len(Xs)):
    X.append(Xs[i-window:i,:])
    Y.append(Ys[i])

# Each time step uses last 'window' to predict the next change
window = 1
Xt = []
Yt = []
for i in range(window, len(Xtest)):
    Xt.append(Xtest[i - window:i, :])
    Yt.append(Ytest[i])

# Reshape data to format accepted by LSTM
X, Y = np.array(X), np.array(Y)
Xtest, Ytest = np.array(Xt), np.array(Yt)
# create and train LSTM model

# Initialize LSTM model
model = Sequential()

model.add(LSTM(512, return_sequences=True,  activation='tanh', input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(8, activation='tanh'))
model.add(Dropout(0.2))
model.add(LSTM(units=1, activation='sigmoid'))
opt = optimizers.Adam(lr=0.0001)
model.compile(optimizer = opt , loss = 'binary_crossentropy', metrics = ['accuracy'])

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
plt.savefig('loss.png')
model.save('model.h5')

# Verify the fit of the model
Yp = model.predict(X, verbose=0)
Yresult = model.predict(Xtest)
print(Yp)

# invert predictions
Yp = s1.inverse_transform(Yp)
Y = s2.inverse_transform([Y])
Yresult = t1.inverse_transform(Yresult)
Ytest = t2.inverse_transform([Ytest])

# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(Y[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(Ytest[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))

# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(train)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(test)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(test)-1, :] = testPredict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(train))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()

