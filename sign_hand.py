import cv2
import numpy as np
import os
from keras.preprocessing import image
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt



import tensorflow as tf
sess = tf.Session(config=tf.ConfigProto(
      allow_soft_placement=True, log_device_placement=True))


folders = ['C:\\Users\\Kasettakorn\\Desktop\\Sign-Language-Digits-Dataset-master\\Dataset\\0',
         'C:\\Users\\Kasettakorn\\Desktop\\Sign-Language-Digits-Dataset-master\\Dataset\\1',
        'C:\\Users\\Kasettakorn\\Desktop\\Sign-Language-Digits-Dataset-master\\Dataset\\2',
        'C:\\Users\\Kasettakorn\\Desktop\\Sign-Language-Digits-Dataset-master\\Dataset\\3',
        'C:\\Users\\Kasettakorn\\Desktop\\Sign-Language-Digits-Dataset-master\\Dataset\\4',
        'C:\\Users\\Kasettakorn\\Desktop\\Sign-Language-Digits-Dataset-master\\Dataset\\5',
        'C:\\Users\\Kasettakorn\\Desktop\\Sign-Language-Digits-Dataset-master\\Dataset\\6',
        'C:\\Users\\Kasettakorn\\Desktop\\Sign-Language-Digits-Dataset-master\\Dataset\\7',
        'C:\\Users\\Kasettakorn\\Desktop\\Sign-Language-Digits-Dataset-master\\Dataset\\8',
        'C:\\Users\\Kasettakorn\\Desktop\\Sign-Language-Digits-Dataset-master\\Dataset\\9']
images = []
labels = []
def load_image_from_folder(folder):
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        img = cv2.resize(img, (100,100))
        img = image.img_to_array(img)
        label = folder.split(os.path.sep)[-1]
        labels.append(label)
        images.append(img)

for folder in folders:
    load_image_from_folder(folder)
x = np.array(images)/255
y = np.array(labels)
y = to_categorical(y, 10)
'''
f, axarr = plt.subplots(1, 5)
f.set_size_inches(16, 6)
for i in range(5):
    img = x[1500+i]
    axarr[i].imshow(img)
plt.show()
'''
(x_train, x_test, y_train, y_test) = train_test_split(x, y, test_size=0.2, random_state=42)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

#Convolution-NN
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Flatten, Dense
from keras.layers import BatchNormalization



cnn = Sequential()

cnn.add(Conv2D(68, (3, 3), input_shape = (100, 100, 3), activation = 'relu'))
cnn.add(MaxPooling2D(pool_size = (2, 2)))

cnn.add(Conv2D(68, (3, 3), activation = 'relu'))
cnn.add(MaxPooling2D(pool_size = (2, 2)))


cnn.add(Conv2D(68, (3, 3), activation = 'relu'))
cnn.add(MaxPooling2D(pool_size = (2, 2)))

cnn.add(Flatten())        
cnn.add(Dense(200, activation='relu'))
cnn.add(Dense(10, activation='softmax'))

cnn.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
print(cnn.summary())

history = cnn.fit(x_train, y_train, epochs=10, batch_size=32)

plt.plot(history.history['loss'])
plt.title('MLP model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()


score = cnn.evaluate(x_test,  y_test, verbose=0)
print(cnn.metrics_names)
print(score)
