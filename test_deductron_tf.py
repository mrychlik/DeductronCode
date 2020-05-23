# Run Tensorflow based training
from deductron_tf import train
import data

def test_tf_training():
    inputs  = data.comb_inputs
    targets = data.comb_targets
    n_memory   = 3                  # Num. of memory cells
    train(inputs,outputs,n_memory)




