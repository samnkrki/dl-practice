#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 09:04:16 2022

@author: samin

setting weights of pretrained model
"""
# example of tending the vgg16 model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
# load model without classifier layers
#model = VGG16(include_top=False, input_shape=(300, 300, 3))
model = VGG16(include_top=False, input_shape=(300, 300, 3), weights='imagenet')
# mark loaded layers as not trainable
for layer in model.layers:
	layer.trainable = False
    
#can also make individual layers trainable as false
#model.get_layer('block1_conv1').trainable = False
#model.get_layer('block1_conv2').trainable = False

# add new classifier layers
# alternative to a flatten layer is an average pooling layer
flat1 = Flatten()(model.layers[-1].output)
class1 = Dense(1024, activation='relu')(flat1)
output = Dense(10, activation='softmax')(class1)
# define new model
model = Model(inputs=model.inputs, outputs=output)
# summarize
model.summary()
# ...