#----------------------------------------------------------------
# File:     run.py
#----------------------------------------------------------------
#
# Author:   Marek Rychlik (rychlik@email.arizona.edu)
# Date:     Tue May 26 07:47:17 2020
# Copying:  (C) Marek Rychlik, 2019. All rights reserved.
# 
#----------------------------------------------------------------
# @brief    Basic deductron tests.

import test_deductron as ded
ded.test_exact_model()
ded.test_large_model_1()
ded.test_comb_model_1()
ded.test_comb_model_2()

# Train a model with Metropolis-Hastings
import test_metropolis_learning as sim_annealing
sim_annealing.test_learning()

# Run Tensorflow based training
import test_deductron_tf as ded_tf
ded_tf.test_tf_training()


