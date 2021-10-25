from utils import loadData

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras import datasets, layers, models


(feature, labels) = loadData()

x_train, x_test, y_train, y_test = train_test_split(
    feature, labels, test_size=0.1)

input_layer = tf.keras.layers.Input([224, 224, 3])


model = models.Sequential([
    layers.Conv2D(filters=32, kernel_size=(3, 3),
                  activation='relu', padding='Same'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])


model.fit(x_train, y_train, batch_size=100, epochs=10)

model.save('mymodel.h5')
