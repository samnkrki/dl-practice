#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 16 16:14:11 2022
https://towardsdatascience.com/implementing-alexnet-cnn-architecture-using-tensorflow-2-0-and-keras-2113e090ad98
Alexnet implementation
@author: samin
"""
import os
import time
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D,AveragePooling2D, Flatten, Dense
from tensorflow.keras import optimizers, losses

from sklearn.metrics import confusion_matrix
import seaborn as sns 
import numpy as np
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#import cv2

#initialize gpu
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)
    
tf.config.list_physical_devices('GPU')
if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")
    

EPOCH = 1
BATCH_SIZE = 20
CLASS_NAMES= ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
CHECKPOINT_PATH='training_1/cp.cpkt'

(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
print(train_labels.shape)
def create_dataset(images, labels):
    return tf.data.Dataset.from_tensor_slices((images, labels))

validation_images, validation_labels = train_images[:5000], train_labels[:5000]
train_images, train_labels = train_images[5000:], train_labels[5000:]

train_ds = create_dataset(train_images, train_labels)
test_ds = create_dataset(test_images, test_labels)
validation_ds = create_dataset(validation_images, validation_labels)

def plot_dataset(dataset, plot_count=5):
    plt.figure(figsize=(20,20))
    for i, (image, label) in enumerate(dataset.take(plot_count)):
        ax = plt.subplot(plot_count,plot_count,i+1)
        plt.imshow(image)
        plt.title(CLASS_NAMES[label.numpy()[0]])
        plt.axis('off')

plot_dataset(train_ds)

def process_images(image, label):
    # Normalize images to have a mean of 0 and standard deviation of 1
    image = tf.image.per_image_standardization(image)
    # Resize images from 32x32 to 277x277
    image = tf.image.resize(image, (227,227))
    return image, label

# get dataset size
train_ds_size = tf.data.experimental.cardinality(train_ds).numpy()
test_ds_size = tf.data.experimental.cardinality(test_ds).numpy()
validation_ds_size = tf.data.experimental.cardinality(validation_ds).numpy()

print("Training data size:", train_ds_size)
print("Test data size:", test_ds_size)
print("Validation data size:", validation_ds_size)
print('train ds', train_ds)

# create dataset by shuffling and batches
'''
def data_generator(data,labels,msg,batch_size=BATCH_SIZE):
    print("startint generator... ",msg)
    while True:
        for i in range(0,labels.shape[0] // batch_size):
            temp_batch_data = data[batch_size*i:batch_size*(i+1)]
            temp_batch_labels = labels[batch_size*i:batch_size*(i+1)]
            batch_data = []
            for j in range(0, len(temp_batch_data)):
                batch_data.append(cv2.resize(temp_batch_data[j],image_size,interpolation=cv2.INTER_CUBIC))
            batch_labels = np_utils.to_categorical(temp_batch_labels,10)
            c = np.array(batch_data).astype('float32')
            a = preprocess_input(c)
            with graph.as_default():
                b = model_without_top.predict(a)
            yield b,batch_labels

'''
train_ds = (train_ds
                  .map(process_images)
                  .shuffle(buffer_size=train_ds_size/100)
                  .batch(batch_size=BATCH_SIZE, drop_remainder=True))
test_ds = (test_ds
                  .map(process_images)
                  .shuffle(buffer_size=test_ds_size/100)
                  .batch(batch_size=BATCH_SIZE, drop_remainder=True))
validation_ds = (validation_ds
                  .map(process_images)
                  .shuffle(buffer_size=validation_ds_size/100)
                  .batch(batch_size=BATCH_SIZE, drop_remainder=True))

def AlexNet():
    model = keras.models.Sequential([
        keras.layers.Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=(227,227,3)),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
        keras.layers.Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
        keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
        keras.layers.Flatten(),
        keras.layers.Dense(4096, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(4096, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(10, activation='softmax')
    ])
    return model

model = AlexNet()
plot_model(model, to_file = 'model.png')

#tensorboard for visualization
root_logdir = os.path.join(os.curdir, "logs\\fit\\")
def get_run_logdir():
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)
run_logdir = get_run_logdir()
tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)

model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.optimizers.SGD(lr=0.001), metrics=['accuracy'])
model.summary()

model.save('outputs/cifar10_alexnet')

model.fit(train_ds,
          epochs=50,
          validation_data=validation_ds,
          validation_freq=1,
          callbacks=[tensorboard_cb])

# tensorboard --logdir logs
'''
# https://www.tensorflow.org/guide/keras/train_and_evaluate
print("Evaluate")
result = model.evaluate(test_ds)
dict(zip(model.metrics_names, result))

from tensorflow.keras.models import load_model
import cv2
import numpy as np

def external_test():
    model = load_model('outputs/cifar10_alexnet')

    model.compile(optimizer=optimizers.SGD(lr=0.01), 
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
    
    img = cv2.imread('./assets/bird.jpg')
    
    img = cv2.resize(img,(227,227))
    img = np.reshape(img,(1,227,227,3))
    print(img/255, 'image is here')
    img = img.astype('float')/255
    #print(img)
    classes = model.predict_classes(img)
    print(model.predict(img))
    print(classes)
    
external_test()
'''