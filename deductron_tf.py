#----------------------------------------------------------------
# File:     deductron_tf.py
#----------------------------------------------------------------
#
# Author:   Marek Rychlik (rychlik@email.arizona.edu)
# Date:     Tue May 26 07:44:16 2020
# Copying:  (C) Marek Rychlik, 2019. All rights reserved.
# 
#----------------------------------------------------------------
# @brief  Implements Deductron training using Tensorflow.

from __future__ import absolute_import, division, print_function

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

## If you are still using Tensorflow v.1 comment out two lines above
## and uncomment the next one

#import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt
import io
from datetime import datetime
from deductron import DeductronBase

class DeductronTf(DeductronBase):
    '''Deductron implementing Tensorflow training.'''
    
    def __init__(self):
        super(DeductronTf, self).__init__(beta = 1, shift = 0)    

    def train(self, inputs, targets,
              n_memory = 3,
              n_steps = 60000,
              loss_threshold = 0.01,
              learning_rate_adam = 0.001):

        input_len = inputs.shape[0];
        output_len = targets.shape[0];

        ################################################################
        #
        #   Define the graph
        #
        ################################################################

        with tf.name_scope("classify_inputs"):
            inputs  = tf.constant(inputs,  tf.float32, name="inputs")
            targets = tf.constant(targets, tf.float32, name="targets")
            n_frames = inputs.shape[1]
            W1 = tf.get_variable("W1", shape = [2*n_memory, input_len],
                                 dtype = tf.float32)
            B1 = tf.get_variable("B1", shape = [2*n_memory, 1],
                                 dtype = tf.float32)
            h = tf.sigmoid(  tf.matmul(W1, inputs) + B1 )
            [left,right] = tf.split(h, num_or_size_splits=2, axis=0)
            prod = tf.multiply(left, right) #Hadamard product
            prod = tf.unstack(prod, axis=1)
            left = tf.unstack(left, axis=1)

        with tf.name_scope("memory"):
            u = tf.zeros([n_memory])
            zlst = [u]
            for t in range(1, n_frames):
            ## Memory
                u = prod[t-1] * u + ( 1.0 - left[t-1])
                zlst.append(u)
                z = tf.stack(zlst, axis = 1)

        with tf.name_scope("output"):
            W2 = tf.get_variable("W2", shape = [output_len, n_memory],
                                 dtype = tf.float32)
            B2 = tf.get_variable("B2", shape = [output_len, 1],
                                 dtype = tf.float32)
            out = 1.0 - tf.sigmoid( tf.matmul(W2, z ) + B2)
            #loss = -tf.reduce_mean(tf.log(out) * targets + tf.log(1.0 - out) * (1.0 - targets))
            diff = out-targets;
            loss1 = tf.reduce_sum(tf.square(diff))
            loss2 = tf.reduce_sum(tf.square(W1))
            loss3 = tf.reduce_sum(tf.square(B1))
            loss4 = tf.reduce_sum(tf.square(W2))
            loss5 = tf.reduce_sum(tf.square(B2))
            eps1 = 0.0001; eps2 = 0.00001
            loss = loss1  + eps1 * (loss2 + loss3) + eps2 * (loss4 + loss5)


        ################################################################
        #
        #   Run the training
        #
        ################################################################

        # Create an optimizer with the desired parameters.
        #opt = tf.train.GradientDescentOptimizer(learning_rate=0.5)
        opt = tf.train.AdamOptimizer(learning_rate=learning_rate_adam)
        #opt = tf.train.AdagradOptimizer(learning_rate=0.5)

        #
        # Run the training
        #
        train = opt.minimize(loss)

        date = datetime.now().isoformat() # Label for summaries
        tf.summary.scalar("loss-" + date, loss)
        write_op = tf.summary.merge_all()

        with tf.Session() as sess:
            writer = tf.summary.FileWriter('logs', sess.graph)
            init = tf.global_variables_initializer()
            sess.run(init)

            print("--------------------------------------------------------")
            print("{:8} | {:10} | {:10}".format('Step','Reg. loss','Real loss'))
            print("--------------------------------------------------------")
            for step in range(n_steps):
                sess.run(train)
                loss_value = sess.run(loss)
                loss1_value = sess.run(loss1)
                if step % 100 == 0:
                    print("{:8d} | {:10.3f} | {:10.3f}".format(step,loss_value,loss1_value))
                summary = sess.run(write_op)
                writer.add_summary(summary, step)
                writer.flush()
                if loss1_value < loss_threshold:
                    print('*** Iteration stopped when loss < ',
                          loss1_value)
                    break

            writer.close()

            self.W1 = W1.eval();
            self.B1 = B1.eval();
            self.W2 = W2.eval();
            self.B2 = B2.eval();

            outputs = out.eval();
            lossval = loss1.eval();

            sess.close()
        tf.reset_default_graph() # Prepare for another run
        return (outputs,lossval)

