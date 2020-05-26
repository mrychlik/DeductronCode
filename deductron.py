#----------------------------------------------------------------
# File:     deductron.py
#----------------------------------------------------------------
#
# Author:   Marek Rychlik (rychlik@email.arizona.edu)
# Date:     Tue May 26 07:40:10 2020
# Copying:  (C) Marek Rychlik, 2019. All rights reserved.
# 
#----------------------------------------------------------------
# 
# @brief    A deductron simulator and examples.
# 
# This simple implementation of a deductron simulator uses numpy in
# class DeductronBase. Subsequently several concrete weight/bias
# combinations are provided as found by Tensorflow (see file
# deductron_tf.py) with varioius training sets.
#

import numpy as np
import copy
import random


class DeductronBase:  
    '''Deductron base. Does not define weights and biases.  Add suitable
    fields W1, B1, W2, B2 to make it work.

    '''
    W1 = None
    B1 = None
    W2 = None
    B2 = None
    out = None

    def __init__(self, beta = 1, shift = 0):
        ''' Sets beta and shift, which are applied to activations
        before standard (rising) sigmoid is applied. '''
        self.beta = beta
        self.shift = shift

    def sigmoid(self, x):
        ''' Implements sigmoid function. '''
        return 1 / (1 + np.exp(- self.beta * (x - self.shift)))


    def __call__(self, input):
        ''' Runs a deductron on input and returns output '''

        #
        # Run input classifier
        #
        h = self.sigmoid(  self.W1 @ input + self.B1)

        #
        # Split neuron outputs into two groups
        #
        left,right = np.split(h, 2)
        n_frames = 1 if len(input.shape) == 1  else input.shape[1]
        n_memory = left.shape[0]

        #
        # Implement V-gate
        #
        prod = left * right
        z = np.zeros((n_memory, n_frames))
        for t in range(1,n_frames):
            z[:,t] = prod[:,t-1] * z[:,t-1] + (1.0 - left[:,t-1])

        #
        # Classify memories
        #
        self.out = 1.0 - self.sigmoid( self.W2 @ z + self.B2 )

        return self

    def loss(self, targets):
        '''Loss - sum of squares of errors.'''
        diff = self.out - targets
        return np.sum(np.square(diff))

    def __str__(self):
        return ("{}:\n"
                "beta: {}\n"
                "shift: {}\n"                
                "W1:\n{}\n"
                "B1:\n{}\n"
                "W2:\n{}\n"
                "B2:\n{}\n").format(self.__class__.__name__,
                                    self.beta,
                                    self.shift,
                                    self.W1,
                                    self.B1,
                                    self.W2,
                                    self.B2)

class WLangDecoderExact(DeductronBase):
    '''Implements exact decoder derived in the white paper. '''

    def __init__(self):
        super(WLangDecoderExact, self).__init__(beta = 10, shift = 0.5)

        self.W1 = np.array([
            #  0,0  1,0,    2,0,   0,1   1,1    2,1
            [  0,     1,      1,     0,    0,    -1  ], # y[0][0][*];
            [  1,     1,      0,    -1,    0,     0  ], # y[0][1][*];
            [  1,     0,      0,    -1,    0,     0  ], # y[0][2][*];
            [  0,     0,      1,     0,    0,    -1  ], # y[0][3][*];
            # 0,0  1,0,    2,0,   0,1   1,1    2,1
            [  1,     1,      0,    -1,    0,     0  ], # y[1][0][*];
            [  0,     1,      1,     0,    0,    -1  ], # y[1][1][*];
            [ -1,     0,      0,     0,    0,     0  ], # y[1][2][*];
            [  0,     0,     -1,     0,    0,     0  ], # y[1][3][*];
            ]).astype(np.float32)

        self.B1 = np.array([
            [1],   [1],    [1],   [1],
            [1],   [1],    [1],   [1],
            ]).astype(np.float32)

        self.W2 = np.array([
            #     0      1     2    3    
            [    -1,     0,   -1,    0  ],
            [     0,    -1,    0,   -1  ],
            ]).astype(np.float32)

        self.B2 = np.array([
            [2],
            [2],
            ]).astype(np.float32)



class WLangDecoderLargeModel1(DeductronBase):  
    ''' Deductron trained on a 'stretched' sample of length 518. '''
    def __init__(self):
        super(WLangDecoderLargeModel1, self).__init__(beta = 10, shift = 0.5)
        self.W1 = np.array([
            [ 1,  1, -1, -1,  1,  1],
            [ 0,  0, -1,  1, -1,  1],
            [-1,  1, -1,  1, -1,  1],
            [-1, -1, -1, -1, -1, -1],
            [ 1, -1, -1,  1,  0, -1],
            [ 0, -1,  1,  0, -1,  0]
            ]).astype(np.float32)

        self.B1 = np.array([
            [2], [1], [0], [3], [1], [0]
            ]).astype(np.float32)

        self.W2 = np.array([
            [ 1, -1,  1],
            [-1,  1,  1]]).astype(np.float32)
        self.B2 = np.array([[1], [1]]).astype(np.float32)
    
class WLangDecoderCombModel1(DeductronBase):  
    '''Deductron trained on combined 'small' and 'stretched' samples of
    total size 29 + 518.

    '''
    def __init__(self):
        super(WLangDecoderCombModel1, self).__init__(beta = 10, shift = 0.5)
        self.W1 = np.array([
            [ 0,  1,  1,  1,  1, -1],
            [-1,  0,  1,  1, -1,  0],
            [ 0, -1,  0, -1,  0, -1],
            [ 1,  1, -1, -1,  0,  1],
            [-1, -1,  0,  0,  0,  1],
            [-1, -1, -1, -1, -1, -1]]).astype(np.float32)
        self.B1 = np.array([[1], [0], [2], [2], [0], [0]]).astype(np.float32)
        self.W2 = np.array([
            [-1,  1, -1],
            [ 1, -1, -1]]).astype(np.float32)
        self.B2 = np.array([[2], [2]]).astype(np.float32)

class WLangDecoderCombModel2(DeductronBase):  
    '''Deductron trained on combined 'small' and 'stretched' samples of
    total size 29 + 518. Obtained using Tensorflow.

    '''
    def __init__(self):
        super(WLangDecoderCombModel2, self).__init__(beta = 1, shift = 0)
        self.W1 = np.array([[ 3.73, -1.17,  2.82, -0.77,  1.71,  4.,  ],
                            [-0.01, -2.16,  4.53,  4.85, -2.14,  2.66 ],
                            [-2.11,  3.37, -1.81,  0.97, -4.08,  1.7  ],
                            [ 4.98,  4.58, -4.49, -4.28,  4.43,  4.93 ],
                            [ 2.88, -1.26, -1.01, -1.42, -0.02,  2.09 ],
                            [ 0.22, -2.01,  0.05, -0.65,  0.12, -1.11 ]]
       ).astype(np.float32)
        self.B1 = np.array([[ 3.66],
                            [ 2.45],
                                [-1.54],
                                [ 5.29],
                                [ 3.45],
                                [-1.63]]).astype(np.float32)
        self.W2 = np.array([
            [-55.81,  47.05,  15.44],
            [ 29.19, -32.,    24.45]]).astype(np.float32)
        self.B2 = np.array([[ 3.42], [12.51]]).astype(np.float32)



class QuantizedDeductron(DeductronBase):
    '''QuantizedDeductron is a Deductron with discrete weights. It implements
    support for Metropolis dynamics (a.k.a. simulated annealing).
    '''

    ADMISSIBLE_WEIGHTS = [-1,0,1]
    ADMISSIBLE_BIASES  = [0,1,2,3,4,5]

    REP_MAX = 8192
    BETA_MAX = 10

    def __init__(self, n_in, n_memory, n_out, beta = 1):
        super(QuantizedDeductron, self).__init__(beta = beta, shift = 0.5)
        self.__create_random_weights(n_in, n_memory, n_out)

    def __create_random_weights(self, n_in, n_memory, n_out):
        self.W1 = np.reshape(
            random.choices(QuantizedDeductron.ADMISSIBLE_WEIGHTS,
                           k = n_in * 2 *n_memory), (2*n_memory, n_in))
        self.B1 = np.reshape(
            random.choices(QuantizedDeductron.ADMISSIBLE_BIASES,
                           k = 2 *n_memory), (2*n_memory, 1))        
        self.W2 = np.reshape(
            random.choices(QuantizedDeductron.ADMISSIBLE_WEIGHTS,
                           k = n_memory * n_out), (n_out, n_memory))        
        self.B2 = np.reshape(
            random.choices(
                QuantizedDeductron.ADMISSIBLE_BIASES,
                k = n_out),
            (n_out, 1))

    def _modify_W1(self):
        m,n = self.W1.shape
        i = random.choice(range(m))
        j = random.choice(range(n))
        old = self.W1[i,j]
        self.W1[i,j] = random.choice(
            QuantizedDeductron.ADMISSIBLE_WEIGHTS)

        return ('W1', (i, j), old)

    def _modify_B1(self):
        i = random.choice(range(self.B1.shape[0]))
        old = self.B1[i,0]
        self.B1[i] = random.choice(
            QuantizedDeductron.ADMISSIBLE_BIASES) - 0.5

        return ('B1', (i), old)

    def _modify_W2(self):
        m,n = self.W2.shape
        i = random.choice(range(m))
        j = random.choice(range(n))
        old = self.W2[i,j]
        self.W2[i,j] = random.choice(
            QuantizedDeductron.ADMISSIBLE_WEIGHTS)

        return ('W2', (i, j) , old)

    def _modify_B2(self):
        i = random.choice(range(self.B2.shape[0]))
        old = self.B2[i,0]
        self.B2[i] = random.choice(
            QuantizedDeductron.ADMISSIBLE_BIASES) - 0.5

        return ('B2', (i), old)

    def modify(self):
        i = random.choice(range(4))
        if i == 0:
            return self._modify_W1()
        elif i == 1:
            return self._modify_B1()            
        elif i == 2:
            return self._modify_W2()            
        elif i == 3:
            return self._modify_B2()            

    def restore(self, mod_data):
        name, idx, old = mod_data
        if name == 'W1':
            i, j = idx
            self.W1[i,j] = old
        elif name == 'B1':
            i = idx
            self.B1[i] = old
        elif name == 'W2':
            i, j = idx
            self.W2[i,j] = old
        elif name == 'B2':
            i = idx
            self.B2[i] = old

    def run_loss(self, inputs, targets):
        return self(inputs).loss(targets)
            
    @staticmethod
    def train(n_memory, inputs, targets, beta_incr = 0.001):
        '''Implements deductron training by simulated annealing.'''
        n_in, _ = inputs.shape
        n_out, _ = targets.shape        

        print("**** Simulated annealing ****")
        net = QuantizedDeductron(n_in, n_memory, n_out, beta = 0)
        E0 = net.run_loss(inputs, targets)
        net_best = copy.deepcopy(net)
        E_best = E0
        print("%10s %10s %10s %10s"
              % ("Iteration", "Loss", "Best Loss", "Inv. Temp."))
        iter = 0
        rep = 0
        while net.beta < QuantizedDeductron.BETA_MAX:
            mod_data = net.modify()
            E1 = net.run_loss(inputs, targets)
            prob = np.exp( -net.beta * (E1 - E0) );
            if  random.uniform(0, 1) < prob:
                rep = 0
                E0 = E1         # Accept
                if  ( iter % 1000 == 0) or ( E0 < E_best ):
                    print("%10.3d %10.3f %10.3f %10.3f"
                          % (iter, E0, E_best, net.beta))
                    
                if E0 < E_best:
                    E_best = E0
                    net_best = copy.deepcopy(net)
            else:
                # Reject, restore weight, update
                rep += 1
                if (rep < 1000 and rep % 100 == 0) or (rep % 1000 == 0):
                    print("Repeats:", rep)

                if rep >= QuantizedDeductron.REP_MAX:
                    print ("Restart on iteration", iter,
                               "repetitions:", rep) 
                    # Restart
                    net = net_best
                    rep = 0
                else:
                    net.restore(mod_data)
            iter += 1
            net.beta += beta_incr
        return (net_best, E_best)
