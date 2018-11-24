import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import tensorflow as tf

folders = ['Datasets\\Images\\auditorium',
'Datasets\\Images\\bakery',
'Datasets\\Images\\bedroom']
#'Datasets\\Images\\bookstore',
#'Datasets\\Images\\concert_hall',
#'Datasets\\Images\\dining_room',
#'Datasets\\Images\\gym',
#'Datasets\\Images\\kitchen',
#Datasets\\Images\\library',
#'Datasets\\Images\\livingroom',
#'Datasets\\Images\\mall',
#'Datasets\\Images\\movietheater',
#'Datasets\\Images\\museum']
images = []
labels = []
def load_image_from_folder(folder):
    print(folder)
    for filename in os.listdir(folder):
        try:
            img = cv2.imread(os.path.join(folder, filename))
            img = cv2.resize(img, (300,300))
            img = image.img_to_array(img)
            label = folder.split(os.path.sep)[-1]
            labels.append(label)
            images.append(img)
            print(os.path.join(folder, filename))
        except:
            print("Skip!!")
            continue

for folder in folders:
    load_image_from_folder(folder)
x = np.array(images)/255
y = np.array(labels)
print(y)
# integer encode
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(y)
print(integer_encoded)
# binary encode
onehot_encoder = OneHotEncoder(sparse=False, categories='auto')
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
y = np.array(onehot_encoded) 
print(y)

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
cnn.add(Conv2D(100, (3, 3), input_shape=(300,300,3), activation='relu'))
cnn.add(MaxPooling2D(pool_size = (3, 3)))

cnn.add(Conv2D(100, (3, 3) ,activation='relu'))
cnn.add(MaxPooling2D(pool_size = (3, 3)))

cnn.add(Conv2D(68, (3, 3) ,activation='relu'))
cnn.add(MaxPooling2D(pool_size = (3, 3)))

cnn.add(Flatten())
cnn.add(Dense(150, activation='relu'))
cnn.add(Dense(3, activation='softmax'))

cnn.compile(loss='mean_squared_error', metrics=['accuracy'], optimizer='adam')
print(cnn.summary())

history = cnn.fit(x_train, y_train, epochs=10, batch_size=32)
plt.plot(history.history['loss'])
plt.title('MLP model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()

test_image = cv2.imread("Datasets\\Images\\bedroom\\Copy_of_tu_98_2_611_25_l.jpg")
test_image = cv2.resize(img, (300,300))
test_image = image.img_to_array(img)
result = cnn.predict(test_image)
print(result)

