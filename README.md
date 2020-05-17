# DeductronCode
Computer codes accompanying the paper "Deductron - A Recurrent Neural Network".
This is a minimalistic implementation. This code should not be used in
any "production" system.


# Pure Python Deductron implementation
In folder deductron one finds a pure Python implementation of the Deductron RNN.
The learning scheme is discrete state space optimization.
The particular optimization scheme is Metropolis-Hastings.


# Deductron implementation using Tensorflow
In folder deductron_tf there is a Python script implementing Deductron RNN
through Tensorflow.


## Tensorflow version 2 supported
The early versions of this code were written for Tensorflow v.1 and they can
be found on the web.

The current version is updated to work with Tensorflow v.2. However, we use
it in compatibility mode with v.1. In principle, one should be able to go
to v.1 quite easily by modifying the top of the script.

