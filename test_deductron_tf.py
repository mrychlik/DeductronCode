# Run Tensorflow based training
import deductron_tf as ded_tf
import data

def test_tf_training():
    inputs  = data.comb_inputs
    targets = data.comb_targets
    n_memory   = 3                  # Num. of memory cells
    n_steps = 10000
    loss_threshold = 0.1
    learning_rate_adam = 0.01
    ded_tf.train(inputs,targets,n_memory,n_steps,loss_threshold, learning_rate_adam)


if __name__ == '__main__':
    test_tf_training()
    


