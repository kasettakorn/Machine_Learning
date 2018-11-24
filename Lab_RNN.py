# Many-to-one LSTM 
import random
import numpy
from numpy import array
from numpy import argmax
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from numpy import array
 
def generate_sequence(length):
    return array([round(random.uniform(0,1), 2) for i in range(length)])

def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back])
    return numpy.array(dataX), numpy.array(dataY)
    
######################################################################################################
look_back = 3
dataset = generate_sequence(30)
X, Y = create_dataset(dataset, look_back)
X = numpy.reshape(X, (X.shape[0], X.shape[1], 1))          #[samples, time_step, n_features]
print(X.shape)
print(Y.shape)
######################################################################################################
model = Sequential()
model.add(LSTM(10, input_shape=(look_back, 1)))           ## 2 cells , 1 dim 
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
print(model.summary())

history = model.fit(X, Y, epochs=50, batch_size=1, verbose=2) #batch_size=1 คือคำนวณ error 1 record และปรับ weight เลย
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()

# One-to-One Model 
import random
import numpy
from numpy import argmax
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense

import matplotlib.pyplot as plt
 
def generate_sequence(length):                            ## Generate a seqeunce of numbers 
    return array([round(random.uniform(0,1), 2) for i in range(length)])

def create_dataset(dataset, look_back):                   ## Build training data X and Y   
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back])
    return numpy.array(dataX), numpy.array(dataY)

#############################################################################################
look_back = 1                                                ### look back step 
dataset = generate_sequence(30)
X, Y = create_dataset(dataset, look_back)                    ###  y(t) = f(x(t))
print(X.shape)
X = numpy.reshape(X, (X.shape[0], look_back, X.shape[1]))    ## LSTM's input shape (n_samples, n_timesteps, n_dims) 
print(X.shape) #28 row 1 timestep 1 column (3D)

################## Define LSTM ##############################################################
model = Sequential()
model.add(LSTM(10, input_shape=(look_back, 1)))                 #[RNN_cells, n_dims]
model.add(Dense(1))                                             #output y   
model.compile(loss='mean_squared_error', optimizer='adam')
print(model.summary())
############################################################################################

history = model.fit(X, Y, epochs=50, batch_size=1, verbose=2)
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()