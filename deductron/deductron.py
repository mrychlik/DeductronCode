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
from deductron_base import *
import data                     # For sample inputs

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

class WLangDecoderCombModel2(deductron_base.DeductronBase):  
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

#
# Sliding windows for string 'XO'
# surrounded by blanks
#
tiny_inputs = np.array([
    [0, 0, 0, 0, 0, 1],
    [0, 0, 1, 0, 1, 0],
    [0, 1, 0, 1, 0, 0],
    [1, 0, 0, 0, 1, 0],
    [0, 1, 0, 0, 0, 1],
    [0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0],
    [1, 0, 0, 0, 1, 0],
    [0, 1, 0, 0, 0, 1],
    [0, 0, 1, 0, 1, 0],
    [0, 1, 0, 1, 0, 0],
    [1, 0, 0, 0, 0, 0],            
    ]).transpose()

tiny_targets = np.array([
    [0, 0],
    [0, 0],
    [0, 0],
    [1, 0],
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 1],
    [0, 0],
    [0, 0],
    ]).transpose()

# Choose one of the neural nets
#nn = WLangDecoderExact()        # Definitely always works
#nn = WLangDecoderLargeModel1()   # Only works on large sample
#nn = WLangDecoderCombModel1()    # Works except for tiny sample
nn = WLangDecoderCombModel2()    # Works for all

n_digits = 4
print()
print("Tiny sample loss:   ", nn(tiny_inputs).loss(
    tiny_targets
    ).round(n_digits))
print("Small sample loss:  ", nn(data.small_inputs).loss(
    data.small_targets
    ).round(n_digits))
print("Large sample loss:  ", nn(data.large_inputs).loss(
    data.large_targets
    ).round(n_digits))
print("Combined sample loss", nn(data.comb_inputs).loss(
    data.comb_targets
    ).round(n_digits))
