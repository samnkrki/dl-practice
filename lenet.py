#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 08:58:37 2022
lenet Implementation

@author: samin
"""

import tensorflow as tf
import tensorflow.keras as keras
from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
from keras import layers
from keras.layers import Conv2D,AveragePooling2D, Flatten, Dense
from keras import optimizers, losses

from sklearn.metrics import confusion_matrix
import seaborn as sns 
import numpy as np


EPOCH = 25
BATCH_SIZE = 128

(x_train, y_train), (x_test, y_test) = mnist.load_data()

def reshape(x_train, x_test):
    x_train=x_train.reshape((x_train.shape[0],x_train.shape[1],x_train.shape[2],1))
    x_test=x_test.reshape((x_test.shape[0],x_test.shape[1],x_test.shape[2],1))
    
    return x_train, x_test

def normalize(x_train, x_test):
    x_train=x_train.astype('float')/255.0
    x_test =x_test.astype('float')/255.0
    return x_train,x_test

def show_input():
    pass

def Lenet(input_shape):
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
