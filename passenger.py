from keras.models import Sequential
from keras.layers import LSTM, Dense
import csv
import numpy as np
import matplotlib.pyplot as plt

def create_dataset(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back)][0][0]
        dataX.append(a)
        dataY.append(dataset[i + look_back][1])

    return np.array(dataX), np.array(dataY)


##############################################################
look_back = 1 # One-to-One
dataset = []
with open('international-airline-passengers.csv', 'r') as f:
  reader = csv.reader(f)
  next(reader)
  for row in reader:
      dataset.append(row)
X, Y = create_dataset(dataset, look_back)

X = np.reshape(X, (X.shape[0], look_back))
X = np.reshape(X, (X.shape[0], look_back, X.shape[1]))


'''
########Define LSTM################
model = Sequential()
model.add(LSTM(10, input_shape=(look_back, 1)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
print(model.summary())
'''
X = [Y[i] for i in range(-1, len(Y), 2)]
Y = [Y[i] for i in range(1, len(Y), 2)]
print(X)
print(Y)
'''
history = model.fit(X, Y, epochs=50, batch_size=1, verbose=2)
plt.plot(history.history['accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()
'''

