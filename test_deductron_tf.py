# Run Tensorflow based training
import deductron_tf as ded_tf
import data

def test_tf_training_fast():
    inputs  = data.tiny_inputs
    targets = data.tiny_targets
    n_memory   = 3            # Num. of memory cells
    n_steps = 5000            # Num. epochs
    loss_threshold = 0.1      # Loss goal
    learning_rate_adam = 0.01 # Initial learning rate for Adam optimizer
    ded_tf.train(inputs,targets,n_memory,n_steps,loss_threshold, learning_rate_adam)

def test_tf_training_slow():
    inputs  = data.comb_inputs
    targets = data.comb_targets
    n_memory   = 3             # Num. of memory cells
    n_steps = 600000           # Num. epochs
    loss_threshold = 0.01      # Loss goal
    learning_rate_adam = 0.001 # Initial learning rate for Adam optimizer
    ded_tf.train(inputs,targets,n_memory,n_steps,loss_threshold, learning_rate_adam)


if __name__ == '__main__':
    test_tf_training_fast()
    


