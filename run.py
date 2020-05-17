# Basic deductron tests
from rychlik.deductron_pkg.test_deductron import *
test_exact_model()
test_large_model_1()
test_comb_model_1()
test_comb_model_2()

# Train a model with Metropolis-Hastings
from rychlik.deductron_pkg.test_metropolis_learning import *
test_learning()

# Run Tensorflow based training
from rychlik.deductron_tf.test_deductron_tf import *
test_learning_tf()
