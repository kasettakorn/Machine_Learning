import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import tensorflow as tf

folders = ['E:\\Git\\Machine_Learning\\Datasets\\Images\\auditorium',
'E:\\Git\\Machine_Learning\\Datasets\\Images\\bakery',
'E:\\Git\\Machine_Learning\\Datasets\\Images\\bedroom']
'''
'E:\\Git\\Machine_Learning\\Datasets\\Images\\bookstore',
'E:\\Git\\Machine_Learning\\Datasets\\Images\\concert_hall',
'E:\\Git\\Machine_Learning\\Datasets\\Images\\dining_room',
'E:\\Git\\Machine_Learning\\Datasets\\Images\\gym',
'E:\\Git\\Machine_Learning\\Datasets\\Images\\kitchen',
'E:\\Git\\Machine_Learning\\Datasets\\Images\\library',
'E:\\Git\\Machine_Learning\\Datasets\\Images\\livingroom',
'E:\\Git\\Machine_Learning\\Datasets\\Images\\mall',
'E:\\Git\\Machine_Learning\\Datasets\\Images\\movietheater',
'E:\\Git\\Machine_Learning\\Datasets\\Images\\museum']
'''
sess = tf.Session()
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
cnn.add(Dense(64, activation='relu'))
cnn.add(Dense(3, activation='softmax'))

cnn.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
print(cnn.summary())

history = cnn.fit(x_train, y_train, epochs=10, batch_size=32)
plt.plot(history.history['loss'])
plt.title('MLP model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()

#Test Image

print(y_test[:20])
plt.imshow(x_test[4])
result = cnn.predict_classes(x_test, batch_size=32, verbose=1)
print(result)

score = cnn.evaluate(x_test, y_test, verbose=0)
print(cnn.metrics_names)
print(score)

# ROC Curve 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.metrics import roc_curve,auc
from scipy import interp
from itertools import cycle


n_classes=3


pred1=cnn.predict(x_test)

### MACRO
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(np.array(pd.get_dummies(y_test))[:, i], np.array(pd.get_dummies(pred1))[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])


all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

lw=2
plt.figure(figsize=(8,5))
plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='green', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--',color='red', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.annotate('Random Guess',(.5,.48),color='red')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic for Naive Bayes - IRIS DATASET')
plt.legend(loc="lower right")
plt.show()