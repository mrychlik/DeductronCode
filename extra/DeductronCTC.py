#----------------------------------------------------------------
# File:     DeductronCTC.py
#----------------------------------------------------------------
#
# Author:   Marek Rychlik (rychlik@email.arizona.edu)
# Date:     Tue May 26 07:37:43 2020
# Copying:  (C) Marek Rychlik, 2019. All rights reserved.
# 
#----------------------------------------------------------------

import tensorflow as tf
import numpy as np

tf.keras.backend.clear_session()  # For easy reset of notebook state.

from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import concatenate


class DeductronCTC(tf.keras.Model):
    def __init__(self, name='DeductronCTC',
                 n_memory = 3,
                 input_len = 6,
                 n_classes = 2,
                 **kwargs):
        super(DeductronCTC, self).__init__(name=name, **kwargs)
        self.n_memory = n_memory
        self.input_len = input_len
        self.n_classes = n_classes
        self.left  = Dense(units = n_memory,activation="sigmoid",use_bias=True)
        self.right = Dense(units = n_memory,activation="sigmoid",use_bias=True)
        self.logit = Dense(units = n_classes, activation="linear",use_bias=True)

    def call(self, inputs, targets):
        n_frames = inputs.shape[0]
        left  = self.left(inputs)
        right = self.right(inputs)
        prod = tf.multiply(left, right) #Hadamard product
        left = tf.unstack(left, axis=0)
        # V-gate layer
        u = tf.zeros(self.n_memory)
        zlst = [u]
        for t in range(1,n_frames):
            # V-gate
            u = prod[t-1] * u + (1-left[t-1])
            zlst.append(u)
        z = tf.stack(zlst,axis=0)
        logits = self.logit(z)
        return logits

if __name__ == '__main__':
    from tiny_data import *
    ded = DeductronCTC(n_memory = 3, input_len = 6, n_classes = 4)
    outputs = ded(tiny_inputs, tiny_targets)
        
