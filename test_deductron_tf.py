# Run Tensorflow based training
import deductron_tf as ded_tf
import data

def test_tf_training():
    inputs  = data.comb_inputs
    targets = data.comb_targets
    n_memory   = 3                  # Num. of memory cells
    ded_tf.train(inputs,outputs,n_memory)




