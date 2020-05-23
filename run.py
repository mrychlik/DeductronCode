# Basic deductron tests
import test_deductron as ded
ded.test_exact_model()
ded.test_large_model_1()
ded.test_comb_model_1()
ded.test_comb_model_2()

# Train a model with Metropolis-Hastings
import test_metropolis_learning as sim_annealing
sim_annealing.test_learning()



# Run Tensorflow based training
import deductron_tf as ded_tf
ded_tf.train()


