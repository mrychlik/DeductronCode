#----------------------------------------------------------------
# File:     deductron_ctc_tf.py
#----------------------------------------------------------------
#
# Author:   Marek Rychlik (rychlik@email.arizona.edu)
# Date:     Sat Jun  6 17:23:22 2020
# Copying:  (C) Marek Rychlik, 2020. All rights reserved.
# 
#----------------------------------------------------------------

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
        output_len = targets.shape[0] + 1;

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
            out = tf.matmul(W2, z ) + B2

            Y = tf.nn.softmax(out)

            lPrime = (2*len(label)+1)*[blank]
            lPrime[1::2] = label

            label,alphabet = _numerical_label(list(labels))
            blank = len(alphabet)
            lPrime = _lPrime(label,blank)
            tsteps = Y.shape[1]

            S = type('S',(), {"blank":2, "tsteps":tsteps, "label":label})

            alpha = np.zeros([S.tsteps,len(lPrime)])
            alpha[0,0] = Y[S.blank, 0];
            alpha[0,1] = Y[lPrime[1], 0]; # Note label[0]==lPrime[1]

            for t in range(1,S.tsteps):
                for s in range(1,len(lPrime)):
                    if s == 1: 
                        tmp = alpha[t-1,s];
                    elif lPrime[s] == S.blank or s == 2 or lPrime[s] == lPrime[s-2]:
                        tmp = alpha[t-1, s] + alpha[t-1,s-1]
                    else:
                        tmp = alpha[t-1, s] + alpha[t-1,s-1] + alpha[t-1, s-2]
                    alpha[t,s] = Y[lPrime[s], t] * tmp;

            p = alpha[S.tsteps-1,len(lPrime)-1]
            if len(lPrime) > 1:
                p = p + alpha[S.tsteps-1,len(lPrime)-2]
                print("p=",p)
            loss = -np.log(p)



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



if __name__ == '__main__':
    tf.reset_default_graph();
    import data

    inputs  = data.tiny_inputs
    targets = data.tiny_targets

    n_memory   = 3            # Num. of memory cells
    n_steps = 5000            # Num. epochs
    loss_threshold = 0.1      # Loss goal
    learning_rate_adam = 0.01 # Initial learning rate for Adam optimizer

    ded = DeductronTf();
    ret = ded.train(inputs,targets,n_memory,n_steps,loss_threshold, learning_rate_adam)
    #outputs = np.round(ret[0],0)
    #err_count = np.sum(np.abs(outputs-targets))
    #print("Error count: {:3.0f}".format(err_count))
    
