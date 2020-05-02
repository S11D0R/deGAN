import os
import numpy as np
import matplotlib.pyplot as plt

# TensorFlow и tf.keras
import tensorflow as tf
from tensorflow import keras

# Вспомогательные библиотеки
import numpy as np
import matplotlib.pyplot as plt

print('TF v.', tf.__version__)

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
depth, height, width = train_images.shape
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
print('\nDepth:', depth)
print('\nHeight:', height)
print('\nWidth:', width)

train_images = train_images / 255.0
test_images = test_images / 255.0

# model = keras.Sequential()
    # keras.layers.Flatten(input_shape=(28, 28)),
    # keras.layers.Dense(256, activation='relu'),
    # keras.layers.Dense(128, activation='relu'),
    # keras.layers.Dense(64, activation='relu'),
    # keras.layers.Dense(10, activation='softmax')
inp = keras.layers.Input(shape=(3, height, width))
conv1 = keras.layers.Convolution2D(32, 3, 3, padding='same', activation='relu', input_shape=(3, 28, 28))(inp)
conv2 = keras.layers.Convolution2D(32, 3, 3, padding='same', activation='relu', input_shape=(3, 28, 28))(conv1)
pool1 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
drop1 = keras.layers.Dropout(0.5)(pool1)
conv3 = keras.layers.Convolution2D(64, 3, 3, padding='same', activation='relu')(drop1)
pool2 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)
drop2 = keras.layers.Dropout(0.5)(pool2)
flat = keras.layers.Flatten()(drop2)
hidd = keras.layers.Dense(512)(flat)
drop3 = keras.layers.Dropout(0.5)(hidd)
out = keras.layers.Dense(10, activation='softmax')(drop3)
model = keras.Model(input=inp, output=out)


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=10, verbose=1, validation_split=0.1)

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('\nТочность на проверочных данных:', test_acc)

predictions = model.predict(test_images)