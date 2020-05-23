# Run Tensorflow based training
from deductron_tf import DeductronTf
import data
import numpy as np

def test_tf_training_tiny_data():
    inputs  = data.tiny_inputs
    targets = data.tiny_targets
    n_memory   = 3            # Num. of memory cells
    n_steps = 5000            # Num. epochs
    loss_threshold = 0.1      # Loss goal
    learning_rate_adam = 0.01 # Initial learning rate for Adam optimizer
    ded = DeductronTf();

    ret = ded.train(inputs,targets,n_memory,n_steps,loss_threshold, learning_rate_adam)
    print("Outputs: {}".format(np.round(ret[0],0)));

def test_tf_training_small_data():
    inputs  = data.small_inputs
    targets = data.small_targets
    n_memory   = 3            # Num. of memory cells
    n_steps = 20000           # Num. epochs
    loss_threshold = 0.1      # Loss goal
    learning_rate_adam = 0.01 # Initial learning rate for Adam optimizer
    ded = DeductronTf();

    ret = ded.train(inputs,targets,n_memory,n_steps,loss_threshold, learning_rate_adam)
    print("Outputs: {}".format(np.round(ret[0],0)));

def test_tf_training_big_data():
    inputs  = data.comb_inputs
    targets = data.comb_targets
    n_memory   = 3             # Num. of memory cells
    n_steps = 600000           # Num. epochs
    loss_threshold = 0.01      # Loss goal
    learning_rate_adam = 0.001 # Initial learning rate for Adam optimizer
    ded = DeductronTf();

    ret = ded.train(inputs,targets,n_memory,n_steps,loss_threshold, learning_rate_adam)
    print("Outputs: {}".format(np.round(ret[0],0)));

if __name__ == '__main__':
    test_tf_training_tiny_data()
    


