#
# @file   deductron_train.py
# @author Marek Rychlik <marek@cannonball.lan>
# @date   Sun Jun 17 10:25:51 2018
# 
# @brief  Implements Deductron training using Tensorflow.
# 
#
#

from __future__ import absolute_import, division, print_function

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import io
from datetime import datetime
import rychlik.deductron_pkg.data as data # Where the training data are

#tf.enable_eager_execution() 
#tf.executing_eagerly()        # => True

# Get training data; should be numpy 2D arrays
#_inputs  = data.small_inputs
#_targets = data.small_targets

_inputs  = data.comb_inputs
_targets = data.comb_targets

# Set parameters for the network
n_memory   = 3                  # Num. of memory cells
input_len  = 6                  # Input frame size
output_len = 2                  # Output frame size

assert(_inputs.shape[0] == input_len)
assert(_targets.shape[0] == output_len)

################################################################
#
#   Define the graph
#
################################################################

def define_graph():
    with tf.name_scope("classify_inputs"):
        inputs  = tf.constant(_inputs,  tf.float32, name="inputs")
        targets = tf.constant(_targets, tf.float32, name="targets")
        n_frames = inputs.shape[1]
        W1 = tf.Variable("W1", shape = [2*n_memory, input_len], dtype = tf.float32)
        B1 = tf.Variable("B1", shape = [2*n_memory, 1], dtype = tf.float32)
        # Inverse temperature
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
        W2 = tf.Variable("W2", shape = [output_len, n_memory], dtype = tf.float32)
        B2 = tf.Variable("B2", shape = [output_len, 1], dtype = tf.float32)
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
    opt = tf.train.AdamOptimizer(learning_rate=0.001)
    #opt = tf.train.AdagradOptimizer(learning_rate=0.5)

    #
    # Run the training
    #
    train = opt.minimize(loss)
    n_steps = 60000                 # Number of steps
    loss_threshold = 0.01           # Stop if loss < this value

    date = datetime.now().isoformat() # Label for summaries
    tf.summary.scalar("loss-" + date, loss)
    write_op = tf.summary.merge_all()

    with tf.Session() as sess:
        writer = tf.summary.FileWriter('logs', sess.graph)
        init = tf.global_variables_initializer()
        sess.run(init)

        for step in range(n_steps):
            #grads_and_vars = opt.compute_gradients(loss)
            #new_grads_and_vars = grads_and_vars
            #train = opt.apply_gradients(grads_and_vars)
            #sess.run(step.assign_add(1))
            sess.run(train)
            loss_value = sess.run(loss)
            loss1_value = sess.run(loss1)
            if step % 100 == 0:
                print("Step: ", step,
                      "Loss: ", loss_value,
                      "Real Loss: ", loss1_value)

            summary = sess.run(write_op)
            writer.add_summary(summary, step)
            writer.flush()
            if loss1_value < loss_threshold:
                print('*** Iteration stopped when loss < ',
                      loss1_value)
                break

        writer.close()

        print("W1:\n{}".format(np.round(W1.eval(),2)))
        print("B1:\n{}".format(np.round(B1.eval(),2)))
        print("W2:\n{}".format(np.round(W2.eval(),2)))
        print("B2:\n{}".format(np.round(B2.eval(),2)))
        print("Outputs: {}".format(np.round(np.transpose(out.eval()),1)))    
        print("Loss: {}".format(loss.eval()))
        print("Real Loss: {}".format(loss1.eval()))

        sess.close()

