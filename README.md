# DeductronCode
Computer codes accompanying the paper "Deductron - A Recurrent Neural Network".
This is a minimalistic implementation. This code should not be used in
any "production" system.

To run several programs exercising the package functionality, all one
needs to do is run the script:

`python3 run.py`

# Pure Python Deductron implementation
In folder rychlik/deductron_pkg one finds a pure Python implementation of the Deductron RNN.
The learning scheme is discrete state space optimization.
The particular optimization scheme is Metropolis-Hastings.

# Deductron implementation using Tensorflow
In folder rychlik/deductron_tf there is a Python script implementing Deductron RNN
through Tensorflow. This folder has a single script.

The run of the script run.py leaves the Tenslorflow logs in folder
logs, which can be analyzed with Tensorboard. Note that Tensorboard
can be also run interactively on one's computer. Tensorboard starts a
Web server which is used to observe the logs, and it produces a
canonical graph of the loss function. This is the way to track
learning progress.

![Tensorboard screenshot](./images/Screenshot from 2020-05-17 14-21-54.png"Tensorboard view")
