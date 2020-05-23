#
# @file   deductron.py
# @author Marek Rychlik <marek@cannonball.lan>
# @date   Sun Jun 17 09:47:17 2018
# 
# @brief  A deductron simulator and examples
# 
# This simple implementation of a deductron simulator
# uses numpy in class DeductronBase. Subsequently
# several concrete weight/bias combinations are provided
# as found by Tensorflow with different training sets.
#

import numpy as np
from . deductron_base import DeductronBase
from . data    import *                # For sample inputs

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


