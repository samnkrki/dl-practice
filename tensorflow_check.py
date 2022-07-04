#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 06:16:35 2022

@author: samin
"""
import tensorflow as tf

print(tf.__version__)
print('Number of GPU available', len(tf.config.list_physical_devices('GPU')))