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
from keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np
from keras import layers
from keras.layers import Conv2D,AveragePooling2D, Flatten, Dense
from keras import optimizers, losses

from sklearn.metrics import confusion_matrix
import seaborn as sns 
import numpy as np


EPOCH = 10
BATCH_SIZE = 20
CLASS_NAMES= ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
print(train_images, train_labels)

validation_images, validation_labels = train_images[:5000], train_labels[:5000]
train_images, train_labels = train_images[5000:], train_labels[5000:]

train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
print(train_ds)
test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
validation_ds = tf.data.Dataset.from_tensor_slices((validation_images, validation_labels))

plt.figure(figsize=(20,20))
for i, (image, label) in enumerate(train_ds.take(5)):
    ax = plt.subplot(5,5,i+1)
    plt.imshow(image)
    plt.title(CLASS_NAMES[label.numpy()[0]])
    plt.axis('off')
    
def process_images(image, label):
    # Normalize images to have a mean of 0 and standard deviation of 1
    image = tf.image.per_image_standardization(image)
    # Resize images from 32x32 to 277x277
    image = tf.image.resize(image, (227,227))
    return image, label

train_ds_size = tf.data.experimental.cardinality(train_ds).numpy()
test_ds_size = tf.data.experimental.cardinality(test_ds).numpy()
validation_ds_size = tf.data.experimental.cardinality(validation_ds).numpy()
print("Training data size:", train_ds_size)
print("Test data size:", test_ds_size)
print("Validation data size:", validation_ds_size)

train_ds = (train_ds
                  .map(process_images)
                  .shuffle(buffer_size=train_ds_size)
                  .batch(batch_size=BATCH_SIZE, drop_remainder=True))
test_ds = (test_ds
                  .map(process_images)
                  .shuffle(buffer_size=train_ds_size)
                  .batch(batch_size=BATCH_SIZE, drop_remainder=True))
validation_ds = (validation_ds
                  .map(process_images)
                  .shuffle(buffer_size=train_ds_size)
                  .batch(batch_size=BATCH_SIZE, drop_remainder=True))

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

#tensorboard for visualization
root_logdir = os.path.join(os.curdir, "logs\\fit\\")
def get_run_logdir():
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)
run_logdir = get_run_logdir()
tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)

model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.optimizers.SGD(lr=0.001), metrics=['accuracy'])
model.summary()

model.fit(train_ds,
          epochs=50,
          validation_data=validation_ds,
          validation_freq=1,
          callbacks=[tensorboard_cb])

# tensorboard --logdir logs
'''
def reshape(train_images, test_images):
    x_train=x_train.reshape((train_images.shape[0],x_train.shape[1],x_train.shape[2],1))
    x_test=x_test.reshape((x_test.shape[0],x_test.shape[1],x_test.shape[2],1))
    
    return x_train, x_test

def normalize(x_train, x_test):
    x_train=x_train.astype('float')/255.0
    x_test =x_test.astype('float')/255.0
    return x_train,x_test

def show_input():
    pass

def Alexnet(input_shape):
    model = keras.Sequential()
    model.add(Conv2D(filters=6, kernel_size=(5,5), activation='relu', input_shape=shape))
    model.add(AveragePooling2D())
    model.add(Conv2D(filters=16, kernel_size=(5,5), activation='relu'))
    model.add(AveragePooling2D())
    model.add(Flatten())
    model.add(Dense(units=120, activation='relu'))
    model.add(Dense(units=84, activation='relu'))
    model.add(Dense(units=10, activation='softmax'))
    
    return model
# this function is called after model is trained, history is the output of model.fit
def summary_history(history):
  plt.figure(figsize = (10,6))
  plt.plot(history.history['accuracy'], color = 'blue', label = 'train')
  plt.plot(history.history['val_accuracy'], color = 'red', label = 'val')
  plt.legend()
  plt.title('Accuracy')
  plt.show()

def plot_confusion_matrix(y_test, y_test_pred):
    #print(y_test[:range_val], y_test_pred[:range_val])
    con_mat = confusion_matrix(y_test, y_test_pred)
    plt.figure(figsize = (8,6))
    sns.heatmap(con_mat, linewidths = 0.1, cmap = 'Greens', linecolor = 'gray', fmt = '.1f', annot = True)
    plt.xlabel('Predicted classes', fontsize = 20)
    plt.ylabel('True classes', fontsize = 20)


print(x_train.shape, y_train.shape)
print(x_test.shape , y_test.shape)

#reshaping
x_train, x_test = reshape(x_train, x_test)

print(x_train.shape, y_train.shape)
print(x_test.shape , y_test.shape)

#normalizing
x_train,x_test = normalize(x_train, x_test)

#plotting
fig = plt.figure(figsize=(10,10))
for i in range(25):
    ax= fig.add_subplot(5, 5, i+1, xticks=[], yticks=[])
    ax.imshow(np.squeeze(x_train[i]), cmap='gray')
    ax.set_title(y_train[i])
    
#check the shape
print(x_train.shape[1:])
shape = x_train.shape[1:]

#create a model
model = Lenet(input_shape=shape)

#compile the model
model.compile(optimizer=optimizers.SGD(lr=0.01), 
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])

model.summary()
x=model.fit(x_train, y_train, epochs=EPOCH, batch_size = BATCH_SIZE, verbose= 2 , validation_split = 0.3)
loss, accuracy= model.evaluate(x_test, y_test, verbose = 0)

print(f'Accuracy: {accuracy*100}')
summary_history(x)

#save model
model.save('./saved_models/lenet-mnist.h5')

# predict labels for the test set
y_test_pred = []
range_val=len(x_test)
for i in range(range_val):
  img = x_test[i]
  #print(img)
  img = img.reshape(1,28,28,1)
  #img = img.astype('float32')
  # one-hot vector output
  vec_p = model.predict(img)
  # determine the lable corresponding to vec_p
  y_p = np.argmax(vec_p)
  print(vec_p, y_p)
  y_test_pred.append(y_p)

y_test_pred = np.asarray(y_test_pred)
plot_confusion_matrix(y_test[:range_val], y_test_pred)
'''