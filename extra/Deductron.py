#----------------------------------------------------------------
# File:     Deductron.py
#----------------------------------------------------------------
#
# Author:   Marek Rychlik (rychlik@email.arizona.edu)
# Date:     Tue May 26 07:18:28 2020
# Copying:  (C) Marek Rychlik, 2019. All rights reserved.
# 
#----------------------------------------------------------------
# @brief    Deductron implementation as a Keras layer.

import tensorflow as tf
import numpy as np

tf.keras.backend.clear_session()  # For easy reset of notebook state.

from tensorflow.keras import layers

class Deductron(layers.Layer):

    def __init__(self, n_memory = 3, input_len = 6, output_len = 2):
        super(Deductron, self).__init__()

        self.n_memory = n_memory
        self.input_len = input_len
        self.output_len = output_len

        W1_init = tf.random_normal_initializer()
        self.W1 = tf.Variable(initial_value = W1_init(shape = (input_len,2*n_memory),
                                                      dtype='float32'),
                              trainable = True);
                              
        B1_init = tf.random_normal_initializer()
        self.B1 = tf.Variable(initial_value = B1_init(shape = (1,2*n_memory),
                                                      dtype = 'float32'),
                              trainable = True);
        W2_init = tf.random_normal_initializer()
        self.W2 = tf.Variable(initial_value = W2_init(shape = (n_memory,output_len),
                                                      dtype='float32'),
                              trainable = True);
        
        B2_init = tf.random_normal_initializer()
        self.B2 = tf.Variable(initial_value = B2_init(shape = (1,output_len),
                                                      dtype = 'float32'),
                              trainable = True);
                                                      
    def call(self, inputs):
        n_frames = inputs.shape[0]
        h = tf.sigmoid( tf.matmul(inputs,self.W1) + self.B1 )
        [left,right] = tf.split(h, num_or_size_splits=2, axis=1)
        prod = tf.multiply(left, right) #Hadamard product
        prod = tf.unstack(prod, axis=0)
        left = tf.unstack(left, axis=0)
        u = tf.zeros([self.n_memory])
        zlst = [u]
        for t in range(1, n_frames):
            ## Memory
            u = prod[t-1] * u + ( 1.0 - left[t-1])
            zlst.append(u)
            z = tf.stack(zlst, axis = 0)
        out =  tf.matmul(z,self.W2) + self.B2
        return out



if __name__ == '__main__':
    from tiny_data import *

    inputs = tiny_inputs
    targets = tiny_targets
        
    ded = Deductron(n_memory = 3, input_len = 6, output_len = 2)
    outputs = ded(inputs)
