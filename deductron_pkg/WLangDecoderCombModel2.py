import numpy as np
from . DeductronBase import DeductronBase

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


