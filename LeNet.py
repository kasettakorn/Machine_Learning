from keras.models import Sequential
from keras.layers import Dense, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.datasets import cifar10
from keras.utils.np_utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt

cnn = Sequential()
cnn.add(Conv2D(6, (5,5), activation='relu', input_shape=(32,32,3)))
cnn.add(MaxPooling2D(pool_size=(2,2), strides=2))
cnn.add(Conv2D(16, (5,5), activation='relu'))
cnn.add(MaxPooling2D(pool_size=(2,2), strides=2))
cnn.add(Flatten())
cnn.add(Dense(120, activation='relu'))
cnn.add(Dense(84, activation='relu'))
cnn.add(Dense(10, activation='softmax'))

sgd = SGD(lr=0.01)
cnn.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=sgd)
print(cnn.summary())

#load dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
x_train /= 255
x_test /= 255

#training
history = cnn.fit(x_train, y_train, epochs=40, verbose=0)
plt.plot(history.history['loss'])
plt.title('Machine Learning แสดงความคลาดเคลื่อนในการจำแนกภาพ 10 Class โดยใช้ LeNet Convolution\n(LeNet Convolution model loss)', fontname='TH SarabunPSK', fontweight="bold", fontsize='15')
plt.ylabel('ค่า error', fontname='TH SarabunPSK', fontweight="bold", fontsize='14')
plt.xlabel('จำนวนการสอนคอมพิวเตอร์ (รอบ)   **จำนวนรูปภาพที่สอน = 50,000 รูป**', fontname='TH SarabunPSK', fontweight="bold", fontsize='14')
plt.show()
