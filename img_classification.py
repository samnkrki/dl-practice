#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 06:38:09 2022
https://lindevs.com/classify-images-of-dogs-and-cats-using-cnn-and-tensorflow-2/
@author: samin
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf

# download dataset
_URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
path_to_zip = tf.keras.utils.get_file(
    'cats_and_dogs.zip', origin=_URL, extract=True)
PATH = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs_filtered')
train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')

#load train and validation set

BATCH_SIZE = 32
IMG_SIZE = (200, 200)
HEIGHT = 200
WIDTH = 200
train_dataset = tf.keras.utils.image_dataset_from_directory(train_dir,
                                                            shuffle=True,
                                                            batch_size=BATCH_SIZE,
                                                            image_size=IMG_SIZE)
validation_dataset = tf.keras.utils.image_dataset_from_directory(validation_dir,
                                                                 shuffle=True,
                                                                 batch_size=BATCH_SIZE,
                                                                 image_size=IMG_SIZE)
val_batches = tf.data.experimental.cardinality(validation_dataset)
test_dataset = validation_dataset.take(val_batches // 5)
validation_dataset = validation_dataset.skip(val_batches // 5)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(HEIGHT, WIDTH, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
 
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
 
trainHistory = model.fit(train_dataset, epochs=10, validation_data=validation_dataset)
 
plt.plot(trainHistory.history['accuracy'])
plt.plot(trainHistory.history['val_accuracy'])
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'])
plt.grid()
plt.show()
 
(loss, accuracy) = model.evaluate(test_dataset)
print(loss)
print(accuracy)
 
model_save_path = 'output/cat_dog_cnn.h5'
model.save(model_save_path)

classNames = ['cat', 'dog']
model = tf.keras.models.load_model(model_save_path)
 
predictions = model.predict(test_dataset.take(8))
 
i = 0
fig, ax = plt.subplots(1, 8)
for image, _ in test_dataset.take(8):
    predictedLabel = int(predictions[i] >= 0.5)
 
    ax[i].axis('off')
    ax[i].set_title(classNames[predictedLabel])
    ax[i].imshow(image[0].numpy().astype("uint8"))
    i += 1
 
plt.show()