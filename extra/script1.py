#----------------------------------------------------------------
# File:     script1.py
#----------------------------------------------------------------
#
# Author:   Marek Rychlik (rychlik@email.arizona.edu)
# Date:     Tue May 26 16:32:12 2020
# Copying:  (C) Marek Rychlik, 2019. All rights reserved.
# 
#----------------------------------------------------------------
import tensorflow as tf
import numpy as np

tf.keras.backend.clear_session()  # For easy reset of notebook state.

from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import concatenate
from keras.utils import plot_model

from tiny_data import *

n_memory  = 3
n_classes = 4
n_samples = tiny_inputs.shape[0]
input_len = tiny_inputs.shape[1];

inputs = Input(shape=(input_len,))
left  = Dense(units = n_memory,activation="sigmoid",use_bias=True)(inputs)
right = Dense(units = n_memory,activation="sigmoid",use_bias=True)(inputs)
prod = tf.multiply(left, right) #Hadamard product
n_memory = left.shape[-1]
u = tf.zeros([n_memory])
zlst = [u]
for t in range(1,n_samples):
    u = prod[t-1] * u + (1-left[t-1])
    zlst.append(u)
    z = tf.stack(zlst,axis=0)
logit = Dense(units = n_classes, activation="linear",use_bias=True)(z)

model = tf.keras.Model(inputs = inputs, outputs = logit)
model.summary()

