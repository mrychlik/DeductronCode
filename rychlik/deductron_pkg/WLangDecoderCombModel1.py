import numpy as np
from . DeductronBase import DeductronBase

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
