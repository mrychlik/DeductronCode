import numpy as np
from . DeductronBase import DeductronBase

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

