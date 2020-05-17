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
from . DeductronBase import DeductronBase
from . data    import *                # For sample inputs

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
