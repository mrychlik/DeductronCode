#----------------------------------------------------------------
# File:     test_deductron_tf.py
#----------------------------------------------------------------
#
# Author:   Marek Rychlik (rychlik@email.arizona.edu)
# Date:     Tue May 26 07:47:35 2020
# Copying:  (C) Marek Rychlik, 2019. All rights reserved.
# 
#----------------------------------------------------------------
# @brief    Run Tensorflow based training on several examples

from deductron_tf import DeductronTf
import data
import numpy as np

def _test_template(inputs,targets,n_memory,n_steps,loss_threshold, learning_rate_adam):
    ded = DeductronTf();
    ret = ded.train(inputs,targets,n_memory,n_steps,loss_threshold, learning_rate_adam)
    outputs = np.round(ret[0],0)
    err_count = np.sum(np.abs(outputs-targets))
    print("Error count: {:3.0f}".format(err_count))


def test_tf_training_tiny_data():
    inputs  = data.tiny_inputs
    targets = data.tiny_targets
    n_memory   = 3            # Num. of memory cells
    n_steps = 5000            # Num. epochs
    loss_threshold = 0.1      # Loss goal
    learning_rate_adam = 0.01 # Initial learning rate for Adam optimizer
    _test_template(inputs,targets,n_memory,n_steps,loss_threshold, learning_rate_adam)

def test_tf_training_small_data():
    inputs  = data.small_inputs
    targets = data.small_targets
    n_memory   = 3            # Num. of memory cells
    n_steps = 20000           # Num. epochs
    loss_threshold = 0.1      # Loss goal
    learning_rate_adam = 0.01 # Initial learning rate for Adam optimizer
    _test_template(inputs,targets,n_memory,n_steps,loss_threshold, learning_rate_adam)


def test_tf_training_big_data():
    inputs  = data.comb_inputs
    targets = data.comb_targets
    n_memory   = 4             # Num. of memory cells
    n_steps = 600000           # Num. epochs
    loss_threshold = 0.01      # Loss goal
    learning_rate_adam = 0.001 # Initial learning rate for Adam optimizer
    _test_template(inputs,targets,n_memory,n_steps,loss_threshold, learning_rate_adam)


if __name__ == '__main__':
    test_tf_training_tiny_data()
    test_tf_training_small_data()
    #test_tf_training_big_data()

